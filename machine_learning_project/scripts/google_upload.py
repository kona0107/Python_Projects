import config
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from httplib2 import Http
from oauth2client import file, client, tools
import os
import glob
import datetime

# OAuth 인증 설정
"""
Google Drive API에 OAuth 인증을 수행하는 코드
- config.GOOGLE_TOKEN_FILE: OAuth 토큰이 저장될 파일 경로
- config.GOOGLE_CREDENTIALS_FILE: OAuth 인증 정보가 저장된 JSON 파일 경로
- creds: OAuth 인증 객체
- credentials.json : 인증 파일 저장명
"""
store = file.Storage(config.GOOGLE_TOKEN_FILE)
creds = store.get()

if not creds or creds.invalid:
    if not os.path.exists(config.GOOGLE_CREDENTIALS_FILE):
        print("`credentials.json` 파일을 찾을 수 없습니다.")
        exit()

    flow = client.flow_from_clientsecrets(config.GOOGLE_CREDENTIALS_FILE, ["https://www.googleapis.com/auth/drive.file"])
    flow.params["access_type"] = "offline"
    flow.params["prompt"] = "consent"

    creds = tools.run_flow(flow, store)

service = build("drive", "v3", http=creds.authorize(Http()))

print("OAuth 인증 완료. 이후에는 자동 로그인됩니다.")

# 오늘 날짜 폴더 생성 (YYYY-MM-DD)
today_date = datetime.datetime.today().strftime("%Y-%m-%d")

# Google Drive에서 날짜별 폴더 찾기 또는 생성
def get_or_create_folder(service, folder_name, parent_folder_id):
    """
    Google Drive에서 특정 부모 폴더 내에 원하는 폴더가 있는지 확인하고, 없으면 생성하는 함수

    매개변수:
    - service: Google Drive API 서비스 객체
    - folder_name (str): 생성하거나 찾을 폴더 이름 (예: '2024-02-11')
    - parent_folder_id (str): 폴더를 생성할 Google Drive 상위 폴더 ID

    반환값:
    - (str) 생성되거나 기존에 존재하는 폴더의 Google Drive ID
    """
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_folder_id}' in parents and trashed=false"
    response = service.files().list(q=query, fields="files(id)").execute()
    folders = response.get("files", [])

    if folders:
        return folders[0]["id"]  # 기존 폴더가 있으면 해당 ID 반환
    else:
        # 새 폴더 생성
        folder_metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_folder_id]
        }
        folder = service.files().create(body=folder_metadata, fields="id").execute()
        return folder["id"]

# 날짜별 폴더 생성 또는 찾기
date_folder_id = get_or_create_folder(service, today_date, config.GOOGLE_FOLDER_ID)

# 업로드할 파일 검색 (result_2024 폴더 내부 파일만)
"""
config.save_path 경로에서 PNG 파일을 찾는 코드
- png_files: 검색된 PNG 파일들의 절대 경로 리스트 (list of str)
"""
png_files = glob.glob(os.path.join(config.save_path, "*.png"))

if not png_files:
    print(f"'{config.save_path}' 내 업로드할 PNG 파일이 없습니다.")
    exit()

print(f"총 {len(png_files)}개의 PNG 파일을 찾았습니다.")

# Google Drive 업로드 (날짜별 폴더에 저장)
"""
파일을 Google Drive의 날짜별 폴더에 업로드하는 코드
- file_path: 로컬에서 업로드할 파일의 전체 경로 (str)
- file_name: Google Drive에 저장될 파일 이름 (str)
- webview_link: 업로드된 파일의 공유 링크 (str)
"""
for file_path in png_files:
    file_name = os.path.basename(file_path)
    print(f"Uploading: {file_name}")

    request_body = {
        "name": file_name,
        "parents": [date_folder_id],  # 날짜별 폴더에 저장
    }

    media = MediaFileUpload(file_path, mimetype="image/png", resumable=True)
    file_info = service.files().create(body=request_body, media_body=media, fields="id,webViewLink").execute()

    webview_link = file_info.get("webViewLink")
    print(f"File Uploaded: {file_name}")
    print(f"Google Drive Link: {webview_link}\n")
