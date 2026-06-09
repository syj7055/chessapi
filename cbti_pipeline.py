import os
import sys
import io
import math
import asyncio   # <-- 추가됨
import statistics
import chess
import chess.engine
import chess.pgn
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from ripser import ripser

# =====================================================================
# 🚨 [중요] Windows + Jupyter 환경에서 NotImplementedError를 방지하는 코드
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
# =====================================================================
class CBTIEvaluator:
    def __init__(self, engine_path="stockfish-windows-x86-64-avx2.exe", depth=15):
        """주피터 노트북 환경에서 엔진을 안전하게 로드합니다."""
        self.engine_path = engine_path
        self.depth = depth
        try:
            # 문제의 setpriority 옵션을 제거하고, 타임아웃만 넉넉하게 15초로 설정
            self.engine = chess.engine.SimpleEngine.popen_uci(
                self.engine_path,
                timeout=15.0 
            )
            print("🎉 [CBTIEvaluator] Stockfish 엔진이 정상적으로 로드되었습니다.")
        except Exception as e:
            raise RuntimeError(f"Engine Load Error: {e}")

    def close(self):
        """평가가 모두 끝나면 엔진을 종료합니다."""
        self.engine.quit()

    def parse_games(self, pgn_text):
        """PGN 텍스트를 게임 객체 리스트로 파싱합니다."""
        pgn_io = io.StringIO(pgn_text)
        games = []
        while True:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break
            games.append(game)
        return games
    
    # ==================== [Axis 1] Memory vs Intuition ====================
    def _load_opening_book(self):
        """
        [내부 메서드] 5개의 TSV 파일을 읽어 모든 오프닝 포지션을 메모리에 Set으로 캐싱합니다.
        - EPD(Extended Position Description)를 사용하여 수순 비틀기(Transposition)를 완벽히 식별합니다.
        """
        self.book_epds = set()
        tsv_files = ['a.tsv', 'b.tsv', 'c.tsv', 'd.tsv', 'e.tsv']
        
        print("📥 오프닝 북(TSV) 데이터를 로드하고 캐싱합니다. (최초 1회 실행)")
        
        for file in tsv_files:
            if not os.path.exists(file):
                print(f"  ⚠️ 경고: {file} 파일을 찾을 수 없어 건너뜁니다.")
                continue
                
            try:
                # 탭(\t)으로 구분된 TSV 파일 로드
                df = pd.read_csv(file, sep='\t')
                
                for pgn_str in df['pgn'].dropna():
                    # TSV의 pgn 문자열을 가상의 파일처럼 읽어 체스 게임 객체로 변환
                    pgn_io = io.StringIO(pgn_str)
                    game = chess.pgn.read_game(pgn_io)
                    
                    if game:
                        board = game.board()
                        self.book_epds.add(board.epd()) # 시작 포지션 저장
                        
                        # 오프닝 수순을 따라가며 모든 도달 포지션을 저장
                        for move in game.mainline_moves():
                            board.push(move)
                            self.book_epds.add(board.epd())
                            
            except Exception as e:
                print(f"  ❌ {file} 로드 중 오류 발생: {e}")
                
        print(f"✅ 오프닝 북 로드 완료: 총 {len(self.book_epds)}개의 고유 오프닝 포지션 캐싱됨.")


    def evaluate_axis1_memory(self, games, target_user):
        """
        1번 축: 오프닝 암기(Memory) vs 직관(Intuition)
        유저의 게임들이 오프닝 북(이론)을 평균적으로 몇 수(Ply)나 따라갔는지 ATD(Average Theoretical Depth)를 반환합니다.
        """
        # 오프닝 북이 아직 로드되지 않았다면 로드
        if not hasattr(self, 'book_epds'):
            self._load_opening_book()

        if not self.book_epds:
            print("⚠️ 오프닝 북 데이터가 없어 1번 축 평가를 진행할 수 없습니다.")
            return 0.0

        theoretical_depths = []

        for game in games:
            # 타겟 유저가 참여한 게임인지 확인
            white_player = game.headers.get("White", "")
            black_player = game.headers.get("Black", "")
            if white_player != target_user and black_player != target_user:
                continue

            board = game.board()
            depth = 0
            
            # 메인라인 수순을 하나씩 따라가며 북에 있는지 확인
            for move in game.mainline_moves():
                board.push(move)
                
                # 현재 포지션이 오프닝 북에 존재하면 깊이(이론수) 1 증가
                if board.epd() in self.book_epds:
                    depth += 1
                else:
                    # 북에 없는 포지션이 나오면(이론에서 벗어나면) 카운팅 종료
                    break
            
            theoretical_depths.append(depth)

        # 타겟 유저의 평균 이론 깊이(ATD) 산출
        if theoretical_depths:
            raw_score = sum(theoretical_depths) / len(theoretical_depths)
        else:
            raw_score = 0.0
            
        return raw_score

    # ==================== [Axis 2] Tactical vs Positional ====================
    def _board_to_vector(self, board):
        """
        [내부 메서드] 체스 보드 상태를 768차원(64칸 * 12기물) One-hot 벡터로 변환합니다.
        (팀원 분이 작성하신 fen_to_vector_local 보다 빠른 Board 객체 직접 참조 방식)
        """
        vector = np.zeros(64 * 12, dtype=int)
        # 백(0~5), 흑(6~11) 매핑
        piece_map = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, 
                     chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
        
        for square, piece in board.piece_map().items():
            color_offset = 0 if piece.color == chess.WHITE else 6
            piece_idx = piece_map[piece.piece_type] + color_offset
            vector[piece_idx * 64 + square] = 1
            
        return vector

    def evaluate_axis2_tactical(self, games, target_user):
        """
        2번 축: 전술(Tactical) vs 전략(Positional) 성향 스코어
        TDA(위상적 데이터 분석)를 활용하여 게임 궤적의 1차원 호몰로지(H1) 수명을 바탕으로 
        전술적 복잡도를 정량화합니다.
        """
        game_scores = []

        for game in games:
            # 타겟 유저가 참여한 게임인지 확인
            white_player = game.headers.get("White", "")
            black_player = game.headers.get("Black", "")
            if white_player != target_user and black_player != target_user:
                continue

            board = game.board()
            vectors = []
            ply_count = 0

            # 메인라인 수순 따라가기
            for move in game.mainline_moves():
                board.push(move)
                ply_count += 1
                
                # 원본 로직 유지: 5수(Ply)마다 벡터 추출
                if ply_count % 5 == 0:
                    vectors.append(self._board_to_vector(board))

            # PCA 및 Ripser를 돌리기 위한 최소 데이터 포인트(5개) 확보 확인
            if len(vectors) < 5:
                continue 

            data = np.array(vectors)
            
            # 차원 축소 (PCA)
            pca = PCA(n_components=min(5, len(data)))
            data_reduced = pca.fit_transform(data)
            
            # 지속성 호몰로지 계산 (TDA)
            try:
                dgms = ripser(data_reduced, maxdim=1)['dgms']
                score = 0.0
                
                # 1차원 구멍(H1)이 존재하는 경우, 생존 시간(Death - Birth)의 합산
                if len(dgms) > 1:
                    h1 = dgms[1]
                    if h1.size > 0:
                        score = np.sum(h1[:, 1] - h1[:, 0])
                
                # 게임 길이에 따른 정규화 (Normalized Complexity)
                normalized_score = score / ply_count
                game_scores.append(normalized_score)
                
            except Exception as e:
                print(f"  ⚠️ 게임 위상 분석 중 오류 발생: {e}")
                continue

        # 타겟 유저의 평균 TDA 전술 복잡도 산출
        if game_scores:
            raw_score = sum(game_scores) / len(game_scores)
        else:
            raw_score = 0.0
            
        return raw_score

    # ==================== [Axis 3] Middlegame vs Endgame ====================
    def _calculate_wp(self, cp):
        cp = max(-10000, min(10000, cp))
        return 1.0 / (1.0 + math.exp(-0.00368208 * cp))

    def _calculate_caps2_accuracy(self, best_cp, actual_cp):
        wp_best = self._calculate_wp(best_cp)
        wp_actual = self._calculate_wp(actual_cp)
        win_prob_lost = max(0.0, wp_best - wp_actual)
        accuracy = 100.0 - (win_prob_lost * 100.0)
        return max(0.0, min(100.0, accuracy))

    def evaluate_axis3_middlegame(self, games, target_user):
        """
        3번 축: 미들게임 vs 엔드게임 (M-Score)
        [source: 3] CAPS2 Accuracy 산출 로직을 통합하였습니다.
        """
        total_mg_sum, total_mg_moves = 0.0, 0
        total_eg_sum, total_eg_moves = 0.0, 0

        for game in games:
            white_player = game.headers.get("White", "")
            black_player = game.headers.get("Black", "")
            if white_player == target_user: player_color = chess.WHITE
            elif black_player == target_user: player_color = chess.BLACK
            else: continue

            board = game.board()
            ply_count = 0
            phase_reached_endgame = False

            for move in game.mainline_moves():
                ply_count += 1
                current_turn = board.turn
                
                # 페이즈 구분
                if ply_count <= 20:
                    current_phase = "Opening"
                else:
                    if not phase_reached_endgame:
                        w_pieces = [p.piece_type for p in board.piece_map().values() if p.color == chess.WHITE and p.piece_type not in (chess.PAWN, chess.KING)]
                        b_pieces = [p.piece_type for p in board.piece_map().values() if p.color == chess.BLACK and p.piece_type not in (chess.PAWN, chess.KING)]
                        cond_a = (len(w_pieces) > 0 and all(p == chess.QUEEN for p in w_pieces)) or \
                                 (len(b_pieces) > 0 and all(p == chess.QUEEN for p in b_pieces))
                        cond_b = sum(1 for p in board.piece_map().values() if p.piece_type in (chess.ROOK, chess.BISHOP, chess.KNIGHT)) <= 2
                        
                        if cond_a or cond_b: 
                            phase_reached_endgame = True
                    current_phase = "Endgame" if phase_reached_endgame else "Middlegame"

                if current_turn == player_color and current_phase in ["Middlegame", "Endgame"]:
                    limit = chess.engine.Limit(depth=self.depth)
                    info_before = self.engine.analyse(board, limit)
                    best_cp = info_before["score"].pov(player_color).score(mate_score=10000)
                    
                    board.push(move)
                    info_after = self.engine.analyse(board, limit)
                    actual_cp = info_after["score"].pov(player_color).score(mate_score=10000)
                    
                    acc = self._calculate_caps2_accuracy(best_cp, actual_cp)
                    
                    if current_phase == "Middlegame":
                        total_mg_sum += acc
                        total_mg_moves += 1
                    else:
                        total_eg_sum += acc
                        total_eg_moves += 1
                else:
                    board.push(move)

        mg_final_acc = (total_mg_sum / total_mg_moves) if total_mg_moves > 0 else 100.0
        eg_final_acc = (total_eg_sum / total_eg_moves) if total_eg_moves > 0 else 100.0
        
        err_mg = max(0.1, 100.0 - mg_final_acc)
        err_eg = max(0.1, 100.0 - eg_final_acc)
        m_score = (err_eg / (err_mg + err_eg)) * 100.0
        
        return m_score

    # ==================== [Axis 4] Aggressive vs Balanced ====================
    def _get_material_advantage(self, board, color):
        score = 0
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        for pt, val in piece_values.items():
            score += len(board.pieces(pt, color)) * val
            score -= len(board.pieces(pt, not color)) * val
        return score

    def _is_imbalanced(self, board):
        for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            if len(board.pieces(pt, chess.WHITE)) != len(board.pieces(pt, chess.BLACK)):
                return True
        return False

    def evaluate_axis4_aggressive(self, games, target_user):
        """
        4번 축: 비대칭/희생 (A-Score)
        [source: 4] TDA 점수 분포 기반 통계 로직을 통합하였습니다.
        """
        results = []
        for game in games:
            white_player = game.headers.get("White", "")
            black_player = game.headers.get("Black", "")
            if white_player == target_user: player_color = chess.WHITE
            elif black_player == target_user: player_color = chess.BLACK
            else: continue

            evals = []
            sacrifices = 0
            imbalances = 0
            mg_moves = 0
            board = game.board()
            ply_count = 0
            phase_reached_endgame = False

            for move in game.mainline_moves():
                ply_count += 1
                if ply_count > 20 and not phase_reached_endgame:
                    cond_b = sum(1 for p in board.piece_map().values() if p.piece_type in (chess.ROOK, chess.BISHOP, chess.KNIGHT)) <= 2
                    if cond_b: phase_reached_endgame = True
                
                current_phase = "Endgame" if phase_reached_endgame else ("Middlegame" if ply_count > 20 else "Opening")

                if current_phase == "Middlegame" and board.turn == player_color:
                    limit = chess.engine.Limit(depth=self.depth)
                    info = self.engine.analyse(board, limit)
                    raw_cp = info["score"].pov(player_color).score(mate_score=10000)
                    eval_cp = max(-1000.0, min(1000.0, float(raw_cp)))
                    mat_adv = self._get_material_advantage(board, player_color)
                    
                    evals.append(eval_cp)
                    mg_moves += 1
                    if mat_adv <= -1 and eval_cp >= -100: sacrifices += 1
                    if self._is_imbalanced(board): imbalances += 1

                board.push(move)

            if mg_moves > 0:
                results.append({
                    'EV': statistics.stdev(evals) if len(evals) > 1 else 0.0,
                    'SR': sacrifices / mg_moves,
                    'IF': imbalances / mg_moves
                })

        df = pd.DataFrame(results)
        if df.empty: return 0.0

        n = len(df)
        E = df['EV'].apply(lambda x: max(0.0, min(1.0, (math.log10(max(x, 0) + 10) - 1.0) / 2.3)))
        S = df['SR'].apply(lambda x: max(0.0, min(1.0, x / 0.30)))
        I = df['IF'].apply(lambda x: max(0.0, min(1.0, x / 1.0)))
        
        points = list(zip(E, S, I))
        distances = sorted([math.sqrt(e**2 + s**2 + i**2) / math.sqrt(3.0) for e, s, i in points], reverse=True)
        
        auc = sum(d * (1.0 / n) for d in distances)
        extreme_cluster_density = sum(1 for d in distances if d >= 0.70) / n
        raw_tda_score = (auc * 70.0) + (extreme_cluster_density * 30.0)
        
        return max(0.0, min(100.0, 100.0 * (raw_tda_score - 30.0) / 20.0))

    # ==================== [Main Pipeline Executor] ====================
    def run_pipeline(self, pgn_text, target_user):
        """파싱된 게임들을 4가지 축 평가 모듈로 일괄 전달합니다."""
        games = self.parse_games(pgn_text)
        if not games:
            return {"error": "파싱할 수 있는 게임이 없습니다."}

        scores = {
            "Axis1_Memory": self.evaluate_axis1_memory(games, target_user),
            "Axis2_Tactical": self.evaluate_axis2_tactical(games, target_user),
            "Axis3_Middlegame": self.evaluate_axis3_middlegame(games, target_user),
            "Axis4_Aggressive": self.evaluate_axis4_aggressive(games, target_user)
        }
        return scores

# ==================== [사용 예시] ====================
if __name__ == "__main__":
    # PGN 텍스트와 유저 ID 예시
    sample_pgn = """[Event "Rated Blitz game"]\n[White "TargetUser"]\n[Black "Opponent"]\n\n1. e4 e5 2. Nf3 Nc6 *"""
    target_username = "TargetUser"

    # Evaluator 초기화 (스톡피쉬 로드)
    evaluator = CBTIEvaluator(engine_path="stockfish-windows-x86-64-avx2.exe")
    
    try:
        # 모든 축의 절대(Raw) 스코어 계산
        final_scores = evaluator.run_pipeline(sample_pgn, target_username)
        print("최종 CBTI Raw 스코어 결과:")
        for axis, score in final_scores.items():
            print(f"- {axis}: {score:.2f}")
    finally:
        evaluator.close() # 엔진 종료 (필수)