import config
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from httplib2 import Http
from oauth2client import file, client, tools
import os
import glob
import datetime

# OAuth 인증 설정
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

# Google Drive에서 날짜별 폴더 찾기
def get_or_create_folder(service, folder_name, parent_folder_id):
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

# 업로드할 파일 검색 (final_2024 폴더 내부 파일만)
png_files = glob.glob(os.path.join(config.save_path, "*.png"))

if not png_files:
    print(f"'{config.save_path}' 내 업로드할 PNG 파일이 없습니다.")
    exit()

print(f"총 {len(png_files)}개의 PNG 파일을 찾았습니다.")

# Google Drive 업로드 (날짜별 폴더에 저장)
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
