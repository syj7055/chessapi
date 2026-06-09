import os
import sys
import stat
import platform
import asyncio
import zipfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from cbti_pipeline import CBTIEvaluator

# =====================================================================
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
# =====================================================================

app = Flask(__name__)
CORS(app)

current_os = platform.system()

# 🧠 [스마트 경로 설정] os.path.join을 쓰면 윈도우(\)와 리눅스(/)를 알아서 맞춰줍니다.
if current_os == 'Windows':
    # 윈도우용 exe 파일이 밖에 있다면 "stockfish-windows-x86-64-avx2.exe" 로 두시고,
    # 만약 폴더 안에 같이 넣으셨다면 아래처럼 사용하세요.
    ENGINE_PATH = "stockfish-windows-x86-64-avx2.exe"
else:
    # Render.com (Linux) 서버: "stockfish/stockfish-ubuntu-x86-64-avx2" 로 자동 변환됨
    ENGINE_PATH = os.path.join("stockfish", "stockfish-ubuntu-x86-64-avx2")

print(f"[{current_os} 환경 감지] 엔진 경로로 '{ENGINE_PATH}'를 사용합니다.")

try:
    # 📦 [압축 해제 로직] 클라우드(Render)에서 stockfish 폴더를 자동으로 복구합니다.
    if current_os != 'Windows':
        # 엔진 경로가 없고, 압축 파일이 존재할 때
        if not os.path.exists(ENGINE_PATH) and os.path.exists("stockfish.zip"):
            print("📦 리눅스 엔진 폴더(stockfish.zip) 압축을 해제합니다...")
            with zipfile.ZipFile("stockfish.zip", 'r') as zip_ref:
                # 현재 폴더(.)에 풀면 원래대로 stockfish 폴더가 짠 하고 나타납니다.
                zip_ref.extractall(".")
            print("✅ 엔진 폴더 압축 해제 완료!")

        # 리눅스 권한 부여 (압축 푼 직후 권한을 주어야 실행됩니다)
        if os.path.exists(ENGINE_PATH):
            st = os.stat(ENGINE_PATH)
            os.chmod(ENGINE_PATH, st.st_mode | stat.S_IEXEC)
            print("✅ 리눅스 엔진 파일에 실행 권한이 부여되었습니다.")

    print("⏳ 엔진 및 오프닝 북을 초기화합니다. 잠시만 기다려주세요...")
    evaluator = CBTIEvaluator(engine_path=ENGINE_PATH)
    print("🚀 [API Server] CBTI 백엔드 서버 준비 완료!")

except Exception as e:
    print(f"❌ 서버 초기화 에러: {e}")
    sys.exit(1)


# =====================================================================
# 🌐 [API 라우트] 프론트엔드에서 기보를 보내면 분석 결과를 반환
# =====================================================================
@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    pgn_text = data.get('pgn')
    username = data.get('username')

    if not pgn_text or not username:
        return jsonify({"error": "PGN 데이터나 유저 이름이 누락되었습니다."}), 400

    print(f"\n📥 [{username}]님의 기보 분석 요청 수신. 계산 시작...")
    try:
        # cbti_pipeline의 4가지 축 계산 실행
        scores = evaluator.run_pipeline(pgn_text, username)
        
        # 만약 파이프라인에서 에러 메시지가 리턴되었다면 그대로 프론트엔드로 전달
        if "error" in scores:
            return jsonify({"error": scores["error"]}), 400
            
        print(f"📤 분석 완료: {scores}")
        return jsonify(scores)

    except Exception as e:
        print(f"❌ 분석 중 치명적 에러 발생: {e}")
        return jsonify({"error": f"서버 내부 오류: {str(e)}"}), 500


# =====================================================================
# 💻 [로컬 테스트용] Render에서는 gunicorn이 실행하므로 이 부분은 무시됨
# =====================================================================
if __name__ == '__main__':
    # 호스트를 0.0.0.0으로 두어 외부 접근도 가능하도록 설정
    app.run(host='0.0.0.0', port=5000, debug=False)