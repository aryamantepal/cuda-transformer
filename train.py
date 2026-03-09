import torch
import lightning as L
from torch.utils.data import TensorDataset, DataLoader
from torch.profiler import schedule, tensorboard_trace_handler
from lightning.pytorch.profilers import PyTorchProfiler

from model import DecoderOnlyTransformer

# ── Config ────────────────────────────────────────────────────────────────────
NUM_TOKENS  = 32000
D_MODEL     = 512
NUM_HEADS   = 8
NUM_LAYERS  = 6
D_FF        = 2048
MAX_LEN     = 512

NUM_SAMPLES = 64        # smaller for local Mac runs
SEQ_LEN     = 128
BATCH_SIZE  = 8
MAX_EPOCHS  = 1
# ─────────────────────────────────────────────────────────────────────────────


def get_accelerator():
    if torch.cuda.is_available():
        return "gpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_dataloader():
    input_data = torch.randint(0, NUM_TOKENS, (NUM_SAMPLES, SEQ_LEN))
    label_data = torch.randint(0, NUM_TOKENS, (NUM_SAMPLES, SEQ_LEN))
    dataset = TensorDataset(input_data, label_data)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def main():
    L.seed_everything(42)
    accel = get_accelerator()
    print(f"Accelerator: {accel}")

    model = DecoderOnlyTransformer(
        num_tokens=NUM_TOKENS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
    )

    dataloader = make_dataloader()

    profiler = PyTorchProfiler(
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=tensorboard_trace_handler("lightning_logs/profiler"),
        record_shapes=True,
        profile_memory=True,
    )

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=accel,
        devices=1,
        profiler=profiler,
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(model, dataloader)

    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("CUDA Memory Summary (post-training)")
        print("=" * 60)
        print(torch.cuda.memory_summary())


if __name__ == "__main__":
    main()
