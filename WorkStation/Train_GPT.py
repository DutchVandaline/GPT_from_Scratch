import torch


def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        logits = model(input_ids, key_padding_mask=attention_mask)

        logits = logits.view(-1, logits.size(-1))
        labels = input_ids.view(-1).to(device)

        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(logits, dim=-1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    # 평균 손실과 정확도 계산
    avg_loss = train_loss / len(dataloader)
    accuracy = correct_preds / total_preds
    return avg_loss, accuracy
