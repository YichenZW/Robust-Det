import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:40960"
import logging
import torch
import json
import argparse
import random
import numpy as np
import datetime
from utils import load_csv, number_h, compute_metrics, histogram_word, to_tensor_dataset
from torch.utils.data import (
    DataLoader,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from torch.optim import AdamW
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
import torch.nn.functional as F

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
current_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
log_filename = f"logs/{current_time}-Finetune-Det.log"
file_handler = logging.FileHandler(log_filename)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default=os.path.join(os.getcwd(), "multi_model_data"),
        type=str,
        help="",
    )
    parser.add_argument(
        "--dataset_name", 
        default="news_gpt-4_t0.7", 
        type=str)
    parser.add_argument(
        "--generate_model_name", 
        type=str, 
        default="gpt-4"
    )
    parser.add_argument(
        "--detect_model_name",
        default="microsoft/deberta-v3-large", 
        type=str,
        help="Model type selected in the list: ",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.getcwd(), "results/cls"),
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_train", default=True, help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", default=True, help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", default=True, help="Whether to run test on the dev set."
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=10,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--seed", type=int, default=82, help="random seed for initialization"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help=""
    )
    args = parser.parse_args()
    return args


args = init_args()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

MODEL_PATH = f"models/{args.detect_model_name.replace('/', '_')}@D_{args.generate_model_name}@G.pt"
logger.info(f"*** Model: {args.detect_model_name.replace('/', '_')}@D_{args.generate_model_name}@G ***")

loss_fn = torch.nn.CrossEntropyLoss()


def train(args, model, train_dataset, eval_dataset):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    logger.info("Total Params: %s", number_h(total_params))
    logger.info("Total Trainable Params: %s", number_h(total_trainable_params))

    train_dataset_len = len(train_dataset)
    t_total = train_dataset_len * args.num_train_epochs / 2
    
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=t_total
    )
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args.batch_size
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=0, last_epoch=-1, verbose=False)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_dataset_len)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    model.zero_grad()
    train_iterator = trange(args.num_train_epochs, desc="Epoch")
    tot_loss, global_step = 0.0, 0
    best_valloss, best_acc = 0.0, 0.0

    for idx, _ in enumerate(train_iterator):
        print({"train/epoch": idx})
        epoch_loss = 0.0

        logger.info(
            "= Learning Rate: {} =".format(
                optimizer.state_dict()["param_groups"][0]["lr"]
            )
        )
        print({"train/lr": optimizer.state_dict()["param_groups"][0]["lr"]})

        with logging_redirect_tqdm():
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[1], "attention_mask": batch[2]}
                outputs = model(**inputs)
                logits = outputs[0]  
                scores = F.softmax(logits, dim=1)[:, 0].squeeze(-1)
                loss = loss_fn(scores, batch[0])
                
                loss.backward()
                tot_loss += loss.item()
                epoch_loss += loss.item()

                epoch_iterator.set_description("loss {}".format(round(epoch_loss / (step + 1) / len(batch[0]), 4)))

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
            # eval each batch
            if args.do_eval:
                res, valloss = eval(
                    args, eval_dataset, model, None, mode="eval", epoch=idx
                )
                if res["acc"] > best_acc:
                    logger.info(
                        "***Best Epoch, Saving Model Into {}***".format(MODEL_PATH)
                    )
                    # best_valloss = valloss
                    best_acc = res["acc"]
                    torch.save(model, MODEL_PATH)

    return tot_loss / global_step


def eval(args, eval_dataset, model, tokenizer, mode, epoch=None):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
    )

    if epoch == None:
        logger.info("***** Running {} *****".format(mode))
    else:
        logger.info("*** Running {} Epoch {} ***".format(mode, epoch))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.batch_size)
    eval_loss, eval_step = 0.0, 0
    preds = None
    with logging_redirect_tqdm():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[1], "attention_mask": batch[2]}
                outputs = model(**inputs)
                scores_softmax = F.softmax(outputs.logits, dim=1)[:, 0].squeeze(-1)
                scores = outputs.logits[:, 0].squeeze(-1)
                loss = loss_fn(scores, batch[0])
            eval_step += 1
            if preds is None:
                preds = scores.detach().cpu().numpy()
                preds_softmax = scores_softmax.detach().cpu().numpy()
                labels = batch[0].detach().cpu().numpy()
            else:
                preds = np.append(preds, scores.detach().cpu().numpy(), axis=0)
                preds_softmax = np.append(preds_softmax, scores_softmax.detach().cpu().numpy(), axis=0)
                labels = np.append(labels, batch[0].detach().cpu().numpy(), axis=0)

    preds = preds.reshape(-1)
    preds_softmax = preds_softmax.reshape(-1)
    histogram_word(preds_softmax, logger=logger)

    logger.info(preds[:10])
    logger.info(preds_softmax[:10])
    logger.info(labels[:10])

    logger.info(preds[-10:])
    logger.info(preds_softmax[-10:])
    logger.info(labels[-10:])
    result = compute_metrics(preds_softmax, labels, num_label=2)

    logger.info("***** Eval results *****")
    for key in result.keys():
        if type(result[key]) is not list:
            print(key, "=", f"{result[key]:.5f}")
            logger.info(f"{key}={result[key]:.5f}")
            print({"val/{}".format(key): result[key]})
    logger.info("  %s = %s", "Logits", str(loss))
    print({"val/Loss": loss})
    return result, loss


def main():
    set_seed(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Current Device: %s", args.device)

    DATASET_PATH = {
        "train": os.path.join(args.data_dir, args.dataset_name + "/" + args.generate_model_name  + "_train.csv"),
        "eval" : os.path.join(args.data_dir, args.dataset_name + "/" + args.generate_model_name + "_val.csv"),
        "test" : os.path.join(args.data_dir, args.dataset_name + "/" + args.generate_model_name + "_test.csv"),
    }

    if args.detect_model_name == "microsoft/deberta-v3-base":
        tokenizer = AutoTokenizer.from_pretrained(
            args.detect_model_name, use_fast=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.detect_model_name,
        )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.detect_model_name, num_labels=2
    ).to(args.device)
    if args.detect_model_name == "openai-gpt":
        tokenizer.pad_token = "pad_token"
        model.config.pad_token_id = 0
    elif tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    train_dataset = load_csv(DATASET_PATH["train"])
    eval_dataset  = load_csv(DATASET_PATH["eval"])
    test_dataset  = load_csv(DATASET_PATH["test"])

    train_dataset = to_tensor_dataset(args, train_dataset, tokenizer)
    eval_dataset  = to_tensor_dataset(args, eval_dataset , tokenizer)
    test_dataset  = to_tensor_dataset(args, test_dataset , tokenizer)
    
    if args.do_train:
        train(args, model, train_dataset, eval_dataset)

    if args.do_test:
        logger.info("Loading Best Model.")
        model = torch.load(MODEL_PATH)
        eval(args, test_dataset, model, tokenizer, mode="test")


if __name__ == "__main__":
    main()
