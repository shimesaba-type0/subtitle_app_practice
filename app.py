# app.py

from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash, jsonify
import os
import whisper
from datetime import timedelta, datetime
import torch
import threading
import uuid
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# 環境変数の読み込み
load_dotenv()

app = Flask(__name__)
CORS(app)  # 必要に応じて設定

# カスタムフィルターの定義
@app.template_filter('datetimeformat')
def datetimeformat(value, format='%Y-%m-%d %H:%M:%S'):
    return datetime.fromtimestamp(value).strftime(format)

# 設定
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')
UPLOAD_FOLDER = 'uploads'
SRT_FOLDER = 'srt'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MODEL_NAME = 'base'  # 使用するWhisperモデル名を 'base' に変更

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SRT_FOLDER'] = SRT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 最大500MB

# ディレクトリの作成
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SRT_FOLDER, exist_ok=True)

# ファイル拡張子の確認
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# SRTフォーマットに変換
def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    formatted_time = str(timedelta(seconds=total_seconds)) + f",{milliseconds:03d}"
    if len(formatted_time.split(":")) == 2:
        formatted_time = "00:" + formatted_time
    return formatted_time

def generate_srt(segments):
    srt = ""
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment['start'])
        end = format_timestamp(segment['end'])
        text = segment['text'].strip()
        srt += f"{i}\n{start} --> {end}\n{text}\n\n"
    return srt

def get_srt_files(page=1, per_page=10):
    try:
        # SRTフォルダ内のファイルリストを取得
        files = [f for f in os.listdir(SRT_FOLDER) if f.lower().endswith('.srt')]
        # ファイルパスと更新時間、サイズのタプルリストを作成
        files_with_info = []
        for f in files:
            filepath = os.path.join(SRT_FOLDER, f)
            mtime = os.path.getmtime(filepath)
            size = os.path.getsize(filepath)
            files_with_info.append({
                'name': f,
                'mtime': mtime,
                'size': size
            })
        # 更新時間でソート（新しい順）
        sorted_files = sorted(files_with_info, key=lambda x: x['mtime'], reverse=True)
        
        # ページネーション
        total = len(sorted_files)
        start = (page - 1) * per_page
        end = start + per_page
        paginated_files = sorted_files[start:end]
        
        # 総ページ数の計算
        total_pages = (total + per_page - 1) // per_page
        
        return paginated_files, total_pages
    except Exception as e:
        print(f"Error retrieving SRT files: {e}")
        return [], 0

# Whisperモデルのロード（GPUを使用）
if torch.cuda.is_available():
    device = "cuda"
    print(f"GPU ({torch.cuda.get_device_name(0)}) を使用しています。")
else:
    device = "cpu"
    print("CPUを使用しています。")

model = whisper.load_model(MODEL_NAME).to(device)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # ファイルが存在するか確認
        if 'file' not in request.files:
            flash("ファイルがありません")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("ファイルが選択されていません")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            original_filename = file.filename
            filename = secure_filename(original_filename)
            unique_id = uuid.uuid4().hex
            base_filename = os.path.splitext(filename)[0]
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            srt_filename = f"{base_filename}_{timestamp}_{unique_id}.srt"
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            srt_path = os.path.join(app.config['SRT_FOLDER'], srt_filename)
            file.save(video_path)

            try:
                # Whisperを使用して文字起こし
                result = model.transcribe(video_path, language="ja")
                segments = result['segments']

                # SRT生成
                srt_content = generate_srt(segments)
                with open(srt_path, 'w', encoding='utf-8') as srt_file:
                    srt_file.write(srt_content)

                # 動画ファイルを削除
                os.remove(video_path)

                flash("字幕生成が完了しました。ダウンロードリンクを選択してください。")
                return redirect(url_for('upload_file'))  # リダイレクトしてGETメソッドで再表示
            except Exception as e:
                flash(f"エラーが発生しました: {str(e)}")
                return redirect(request.url)
        else:
            flash("許可されていないファイル形式です")
            return redirect(request.url)
    
    # GETメソッド時にページ番号を取得（デフォルトは1）
    page = request.args.get('page', 1, type=int)
    per_page = 10  # 1ページあたりのアイテム数
    
    # ページネーション対応のSRTファイルリストを取得
    srt_files, total_pages = get_srt_files(page=page, per_page=per_page)
    
    return render_template('index.html', srt_files=srt_files, page=page, total_pages=total_pages)

@app.route('/srt/<filename>', methods=['GET'])
def srt_file(filename):
    return send_from_directory(app.config['SRT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

