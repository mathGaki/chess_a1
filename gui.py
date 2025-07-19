import sys
import os
import time
import multiprocessing as mp
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont
from PyQt5.QtCore import QTimer, Qt, QRectF, QRect

# GUI 설정
NUM_BOARDS = 64  # 64개 워커를 위한 보드 수
BOARD_SIZE = 160 # 각 체스판의 픽셀 크기 (64개 보드를 위해 더 축소)
PIECE_SIZE = BOARD_SIZE // 8

class ChessBoardWidget(QWidget):
    """개별 체스판을 그리는 위젯"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(BOARD_SIZE, BOARD_SIZE)
        self.board_state = [' '] * 64 # 64칸의 기물 정보 (FEN 기반)
        self.piece_pixmaps = self._load_piece_images()

    def _load_piece_images(self):
        pixmaps = {}
        pieces = {'P': 'wp', 'N': 'wn', 'B': 'wb', 'R': 'wr', 'Q': 'wq', 'K': 'wk',
                  'p': 'bp', 'n': 'bn', 'b': 'bb', 'r': 'br', 'q': 'bq', 'k': 'bk'}
        
        # 여러 가능한 이미지 경로를 시도
        possible_dirs = [
            'src/ai/pieces',
            'src\\ai\\pieces',
            'ai/pieces',
            'pieces',
            '.',
            os.path.join(os.path.dirname(__file__), 'pieces')
        ]
        
        image_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path) and any(os.path.exists(os.path.join(dir_path, f"{filename}.png")) for filename in pieces.values()):
                image_dir = dir_path
                break
        
        if not image_dir:
            print("Warning: No piece images directory found")
            return pixmaps
        
        for piece, filename in pieces.items():
            path = os.path.join(image_dir, f"{filename}.png")
            if os.path.exists(path):
                pixmaps[piece] = QPixmap(path)
            else:
                print(f"Warning: Piece image not found: {path}")
        
        return pixmaps

    def update_from_fen(self, fen_board_part):
        """FEN의 보드 부분 문자열로 보드 상태 업데이트"""
        self.board_state = [' '] * 64
        rank, file = 0, 0
        for char in fen_board_part:
            if char == '/':
                rank += 1
                file = 0
            elif char.isdigit():
                file += int(char)
            else:
                sq_idx = rank * 8 + file
                self.board_state[sq_idx] = char
                file += 1
        
        self.update() # 위젯을 다시 그리도록 요청

    def paintEvent(self, event):
        painter = QPainter(self)
        
        # 체스판 칸 그리기 (클래식 나무색)
        for rank in range(8):
            for file in range(8):
                x, y = file * PIECE_SIZE, rank * PIECE_SIZE
                if (rank + file) % 2 == 0:
                    color = QColor(240, 217, 181)  # 밝은 나무색 (F0D9B5)
                else:
                    color = QColor(181, 136, 99)   # 어두운 나무색 (B58863)
                painter.fillRect(x, y, PIECE_SIZE, PIECE_SIZE, color)

        # 기물 그리기
        for sq_idx, piece in enumerate(self.board_state):
            if piece != ' ' and piece in self.piece_pixmaps:
                pixmap = self.piece_pixmaps[piece]
                rank, file = sq_idx // 8, sq_idx % 8
                x, y = file * PIECE_SIZE, rank * PIECE_SIZE
                
                target_rect = QRectF(x, y, PIECE_SIZE, PIECE_SIZE)
                source_rect = QRectF(pixmap.rect())  # QRect를 QRectF로 변환
                
                painter.drawPixmap(target_rect, pixmap, source_rect)

class MainWindow(QMainWindow):
    """64개의 체스판을 보여주는 메인 윈도우"""
    def __init__(self, gui_queue):
        super().__init__()
        self.gui_queue = gui_queue
        self.setWindowTitle("AlphaZero Chess - Training Monitor (64 Workers)")
        self.setGeometry(100, 100, 1400, 1400)  # 64개 보드를 위해 정사각형 배치 (8x8)
        
        self.init_ui()
        self.init_timer()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)  # 체스판만 표시
        
        # 체스판들 (8x8 배치)
        board_widget = QWidget()
        board_layout = QGridLayout(board_widget)
        board_layout.setSpacing(2)  # 보드 간 간격 더 축소
        self.boards = []
        for i in range(NUM_BOARDS):
            board = ChessBoardWidget()
            row, col = i // 8, i % 8  # 8x8 배치 (8열 8행)
            board_layout.addWidget(board, row, col)
            self.boards.append(board)
        
        # 메인 레이아웃에 체스판만 추가
        main_layout.addWidget(board_widget)

    def init_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_boards)
        self.timer.start(50) # 0.05초(50ms)마다 업데이트 - 더 빠른 반응성

    def update_boards(self):
        while not self.gui_queue.empty():
            try:
                data = self.gui_queue.get_nowait()
                
                # 이전 방식 호환성 (worker_id, fen_board 튜플)
                if isinstance(data, tuple) and len(data) == 2:
                    worker_id, fen_board = data
                    if 0 <= worker_id < NUM_BOARDS:
                        self.boards[worker_id].update_from_fen(fen_board)
                
                # 새로운 방식 (딕셔너리) - FEN 업데이트만 처리
                elif isinstance(data, dict) and data.get('type') == 'fen':
                    board_idx = data.get('board_id', 0) % NUM_BOARDS
                    fen_board_part = data['fen'].split()[0]
                    self.boards[board_idx].update_from_fen(fen_board_part)
                            
            except Exception as e:
                pass  # 큐가 비어있거나 다른 오류 무시

def start_gui(gui_queue):
    """GUI 시작 함수"""
    app = QApplication(sys.argv)
    window = MainWindow(gui_queue)
    window.show()
    app.exec_()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    gui_queue = mp.Queue()
    
    # 테스트용 더미 데이터
    initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    for i in range(NUM_BOARDS):
        gui_queue.put((i, initial_fen))
    
    start_gui(gui_queue)
