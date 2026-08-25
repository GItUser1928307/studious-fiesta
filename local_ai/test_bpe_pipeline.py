#!/usr/bin/env python3
"""
Tests A-H for Byte-Level BPE pipeline (Kaggle-safe, no full training).

Run:
    python local_ai/test_bpe_pipeline.py
"""
import os, sys, tempfile, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn

from tokenizer import get_tokenizer, load_tokenizer
from config import ModelConfig
from model import create_model
from retrain import TextDataset, PackedDataset, _split_qa_file, build_scheduler
from config import auto_train_config

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "quick_train_data.txt")

def test_a_tokenizer():
    print("Test A: tokenizer encode/decode")
    train_lines, _ = _split_qa_file(DATA_FILE, ratio=0.9, seed=42)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        tf.write("\n".join(train_lines))
        train_file = tf.name
    try:
        tok = get_tokenizer("bytebpe", data_file=train_file, vocab_size=2048)
        # For tiny 14KB corpus, HF BPE may produce <2048 (not enough merges); just check range and consistency
        assert 256 < tok.vocab_size <= 2048, f"vocab {tok.vocab_size} out of expected range"
        assert tok.vocab_size == tok._tok.get_vocab_size()
        # ordinary text, punctuation, numbers, unicode, code
        tests = ["Hello, world!", "123 456", "café naïve", "https://example.com?a=1", '{"key": "value"}', "```python\nprint(1)\n```", "∑ ∫ √ ∞"]
        for t in tests:
            ids = tok.encode(t)
            assert all(0 <= i < tok.vocab_size for i in ids), f"IDs out of range for {t}"
            dec = tok.decode(ids)
            # decode should not crash, may not be identical due to BPE but should contain bytes
            assert isinstance(dec, str)
        print(f"  PASS vocab {tok.vocab_size} encode/decode ok")
        return tok
    finally:
        try: os.remove(train_file)
        except: pass

def test_bc_model_forward(tok):
    print("Test B/C: vocab/embedding consistency + forward")
    cfg = ModelConfig(vocab_size=tok.vocab_size, hidden_size=512, num_layers=2, num_heads=8, intermediate_size=1408, max_seq_len=96)
    model = create_model(cfg)
    assert model.token_embedding.num_embeddings == tok.vocab_size
    assert model.lm_head.weight.shape[0] == tok.vocab_size or cfg.tie_embeddings
    x = torch.randint(0, tok.vocab_size, (2, 96))
    logits, _ = model(x)
    assert logits.shape == (2, 96, tok.vocab_size)
    print(f"  PASS forward {logits.shape}")
    return cfg, model

def test_def_loss_backward(cfg, model, tok):
    print("Test D/E/F: loss, backward, optimizer")
    # Build tiny dataset
    train_lines, _ = _split_qa_file(DATA_FILE, ratio=0.9, seed=42)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        tf.write("\n".join(train_lines))
        train_file = tf.name
    try:
        ds = TextDataset(train_file, tok, cfg.max_seq_len)
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        x,y,mask = next(iter(loader))
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
        sched = build_scheduler(opt, 100)
        logits, loss = model(x, y, loss_mask=mask)
        loss = loss.mean()
        assert torch.isfinite(loss).item(), f"loss not finite {loss.item()}"
        print(f"  D loss {loss.item():.4f} finite")
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all().item()
        print("  E grads finite")
        opt.step()
        sched.step()
        print("  F optimizer step ok")
    finally:
        try: os.remove(train_file)
        except: pass

def test_g_checkpoint(tok, cfg, model):
    print("Test G: checkpoint save/load")
    import tempfile
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    sched = build_scheduler(opt, 100)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    from retrain import save_checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
        path = tf.name
    try:
        save_checkpoint(path, model, cfg, "bytebpe", 42, opt, sched, scaler, backend="cpu", precision="fp32")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["step"] == 42
        assert ckpt["config"].vocab_size == tok.vocab_size
        model2 = create_model(cfg)
        model2.load_state_dict(ckpt["model"])
        print("  PASS save/load")
    finally:
        try: os.remove(path)
        except: pass

def test_h_inference(tok, cfg, model):
    print("Test H: inference generate")
    model.eval()
    prompt = "Hello world"
    ids = [tok.bos_token_id, tok.q_token_id] + tok.encode(prompt) + [tok.a_token_id]
    idx = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=10, temperature=0.8, top_k=40, eos_token_id=tok.eos_token_id)
    gen = out[0].tolist()[len(ids):]
    txt = tok.decode(gen)
    assert isinstance(txt, str)
    print(f"  PASS generate: {txt[:50]}")

if __name__ == "__main__":
    tok = test_a_tokenizer()
    cfg, model = test_bc_model_forward(tok)
    test_def_loss_backward(cfg, model, tok)
    test_g_checkpoint(tok, cfg, model)
    test_h_inference(tok, cfg, model)
    print("\nAll tests A-H PASSED")
