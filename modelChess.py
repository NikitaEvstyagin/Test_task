import pandas as pd
import numpy as np
import torch
import chess
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim




df = pd.read_csv('fens_training_set.csv', names=['fen', 'move'])


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Каналы: белые фигуры (6), чёрные фигуры (6)
    Порядок: P, N, B, R, Q, K
    
    :param board: fen
    :return: доску в тензор формы (12, 8, 8)
    """

    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    for i, piece_type in enumerate(pieces):
        # Белые
        for sq in board.pieces(piece_type, chess.WHITE):
            row, col = divmod(sq, 8)
            tensor[i][7 - row][col] = 1.0
        # Чёрные
        for sq in board.pieces(piece_type, chess.BLACK):
            row, col = divmod(sq, 8)
            tensor[i + 6][7 - row][col] = 1.0

    return torch.from_numpy(tensor)

def move_to_index(move: chess.Move) -> int:
    """
    Превращение пешки: 4 типа (N, B, R, Q) × 8 направлений.
    Все превращения - в ферзя кодирую как обычный ход.
    :param move: Ход
    :return: индекс хода в упрощённой 4096-мерной системе
    """    
    return move.from_square * 64 + move.to_square



def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """
    
    :param idx: индекс хода в упрощённой 4096-мерной системе
    :param board: Тензор формы (12, 8, 8)
    :return: Восстановленный ход из индекса в 4096-мерном пространстве
    """
    from_sq = idx // 64
    to_sq = idx % 64

    to_row = to_sq // 8
    piece = board.piece_at(from_sq)
    promotion = None
    if piece and piece.piece_type == chess.PAWN:
        if (piece.color == chess.WHITE and to_row == 0) or (piece.color == chess.BLACK and to_row == 7):
            promotion = chess.QUEEN
    return chess.Move(from_sq, to_sq, promotion=promotion)

def is_valid_fen(fen: str) -> bool:
    try:
        chess.Board(fen)
        return True
    except ValueError:
        return False

def is_valid_move(fen: str, move_uci: str) -> bool:
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        return move in board.legal_moves
    except (ValueError, AssertionError):
        return False



#Простая CNN с 2 скрытыми слоями, функцией активации ReLu
class ChessNet(nn.Module):
    def __init__(self, num_outputs=4096):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512) 
        self.fc2 = nn.Linear(512, num_outputs)
        self.dropout = nn.Dropout(0.5)
        self.dropout_conv = nn.Dropout2d(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout_conv(x) 
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
#DataSet
class ChessDataset(Dataset):    
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fen = self.df.loc[idx, 'fen']
        move_uci = self.df.loc[idx, 'move']
        move = chess.Move.from_uci(move_uci)
        move_idx = move_to_index(move)
        board_tensor = board_to_tensor(chess.Board(fen))
        return board_tensor, move_idx, fen


def train_model(model, train_loader, device, epochs=5, lr=1e-3):
    """
    
    :param model:  Обучаемая модель
    :param train_loader: Загрузчик тренировочных данных.
    :param device: 
    :param epochs: Число эпох обучения.
    :param lr: Шаг обучения.
    :return: 
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for boards, targets, _ in train_loader:
            boards, targets = boards.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        
def evaluate_model(model, test_loader, device, top_k=1) -> float:
    """
    
    :param model: Обученная модель.
    :param test_loader:  Загрузчик тестовых данных.
    :param device: 
    :param top_k:  Количество лучших предсказаний для учёта
    :return: Доля корректно предсказанных ходов 
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for boards, targets, fens in test_loader:
            boards = boards.to(device)
            logits = model(boards)
            for i, fen in enumerate(fens):
                board = chess.Board(fen)
                legal_mask = torch.full((4096,), -1e9, dtype=torch.float32)
                for move in board.legal_moves:
                    idx = move.from_square * 64 + move.to_square
                    legal_mask[idx] = 0.0
                masked_logits = logits[i].cpu() + legal_mask
                if top_k == 1:
                    pred = masked_logits.argmax().item()
                    if pred == targets[i].item():
                        correct += 1
                else:
                    top_preds = masked_logits.topk(top_k).indices
                    if targets[i].item() in top_preds:
                        correct += 1
                total += 1
    return correct / total


def predict_move(model, fen: str, device, top_k=5) -> list[tuple[str, float]]:
    """
    Предсказывает наиболее вероятные ходы для заданной позиции.

    :param model: Обученная модель.
    :param fen: Позиция в формате FEN.
    :param device: 
    :param top_k: Количество возвращаемых ходов.
    :return: Список пар (ход_в_UCI, вероятность).
    """
    board = chess.Board(fen)
    x = board_to_tensor(board).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)[0].cpu()
    legal_mask = torch.full((4096,), -1e9, dtype=torch.float32)
    for move in board.legal_moves:
        idx = move.from_square * 64 + move.to_square
        legal_mask[idx] = 0.0
    masked_logits = logits + legal_mask
    probs = F.softmax(masked_logits, dim=0)
    top_probs, top_indices = torch.topk(probs, min(top_k, len(list(board.legal_moves))))
    results = []
    for prob, idx in zip(top_probs, top_indices):
        move = index_to_move(idx.item(), board)
        if move in board.legal_moves:
            results.append((move.uci(), prob.item()))
        else:
            for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                alt_move = chess.Move(move.from_square, move.to_square, promotion=promo)
                if alt_move in board.legal_moves:
                    results.append((alt_move.uci(), prob.item()))
                    break
    return results


def main():
    # Пути и параметры
    data_path = 'fens_training_set.csv'
    batch_size = 64
    epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    df = pd.read_csv('fens_training_set.csv', names=['fen', 'move'])
    print(f"До фильтрации: {len(df)} строк")

    mask = df.apply(lambda row: is_valid_fen(row['fen']) and is_valid_move(row['fen'], row['move']), axis=1)
    df = df[mask].reset_index(drop=True)

    print(f"После фильтрации: {len(df)} строк")
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


    # Датасеты и загрузчики
    train_dataset = ChessDataset(train_df)
    test_dataset = ChessDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Модель
    model = ChessNet(num_outputs=4096).to(device)

    # Обучение
    print("Обучение")
    train_model(model, train_loader, device, epochs=epochs)

    # Оценка
    print("Оценка на тесте:")
    top1 = evaluate_model(model, test_loader, device, top_k=1)
    top3 = evaluate_model(model, test_loader, device, top_k=3)
    print(f"Test Top-1 Accuracy: {top1:.4f}")
    print(f"Test Top-3 Accuracy: {top3:.4f}")


if __name__ == "__main__":
    main()