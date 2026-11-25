#!/usr/bin/env python3

import json
import os
import subprocess
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def create_dummy_tokenizer(tokenizer_dir: Path) -> None:
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    vocab = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
    vocab.update({f"dummy_{i}": i + 4 for i in range(32764)})

    tokenizer = Tokenizer(BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevel()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))

    (tokenizer_dir / "special_tokens_map.json").write_text(
        json.dumps({"unk_token": "<unk>", "pad_token": "<pad>", "bos_token": "<bos>", "eos_token": "<eos>"}),
        encoding="utf-8",
    )
    (tokenizer_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 1024}),
        encoding="utf-8",
    )


def create_dummy_data(path: Path, line_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = " ".join(str(4 + (i % 32764)) for i in range(1024))
    with path.open("w", encoding="utf-8") as handle:
        for _ in range(line_count):
            handle.write(line + "\n")


def main() -> None:
    num_gpus = env_int("NUM_GPUS", 1)
    devices = os.environ.get("DEVICES", ",".join(str(index) for index in range(num_gpus)))
    master_port = os.environ.get("MASTER_PORT", "29510")

    data_dir = Path(os.environ.get("TEST_DATA_DIR", str(PROJECT_ROOT / "test_data")))
    tokenizer_dir = Path(os.environ.get("TEST_TOKENIZER_DIR", str(data_dir / "dummy_tokenizer")))
    train_file = Path(os.environ.get("TEST_TRAIN_FILE", str(data_dir / "dummy_train.packed.txt")))
    eval_file = Path(os.environ.get("TEST_EVAL_FILE", str(data_dir / "dummy_eval.packed.txt")))

    data_dir.mkdir(parents=True, exist_ok=True)

    if not (tokenizer_dir / "tokenizer.json").is_file():
        create_dummy_tokenizer(tokenizer_dir)
    if not train_file.is_file():
        create_dummy_data(train_file, 512)
    if not eval_file.is_file():
        create_dummy_data(eval_file, 16)

    run_name = os.environ.get("RUN_NAME", "test_350m_dummy")
    save_dir = Path(os.environ.get("SAVE_DIR", str(PROJECT_ROOT / "checkpoints" / run_name)))

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": devices,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8"),
            "NCCL_TIMEOUT": os.environ.get("NCCL_TIMEOUT", "1800"),
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": os.environ.get("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "300"),
            "WANDB_MODE": os.environ.get("WANDB_MODE", "offline"),
        }
    )

    command = [
        str(PYTHON),
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(num_gpus),
        "--master_port",
        master_port,
        "torchrun_main.py",
        "--model_config",
        "configs/llama_350m.json",
        "--lr",
        "1e-3",
        "--batch_size",
        "16",
        "--total_batch_size",
        "512",
        "--num_epochs",
        "1",
        "--warmup_steps_ratio",
        "0.1",
        "--weight_decay",
        "0",
        "--dtype",
        "bfloat16",
        "--eval_every",
        "500",
        "--optimizer",
        "adam",
        "--grad_clipping",
        "1.0",
        "--max_length",
        "1024",
        "--workers",
        "0",
        "--run_name",
        run_name,
        "--save_dir",
        str(save_dir),
        "--dataset_name",
        "fineweb_eng",
        "--monolingual-dataset",
        "fineweb_eng",
        "--tokenizer_path",
        str(tokenizer_dir),
        "--tokenizer_type",
        "bpe",
        "--tokenizer_vocabulary",
        "32768",
        "--train_file",
        str(train_file),
        "--eval_file",
        str(eval_file),
    ]

    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
