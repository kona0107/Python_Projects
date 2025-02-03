from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# Chrome 브라우저 설정
options = webdriver.ChromeOptions()
options.add_experimental_option('detach', True)
options.add_experimental_option('prefs', {
    "download.default_directory": "C:\\사용자\\EDAM\\다운로드",  # 다운로드 폴더 경로 설정
    "download.prompt_for_download": False,  # 다운로드 대화 상자 비활성화
})
options.add_argument('--ignore-certificate-errors')  # SSL 인증 오류 무시
options.add_argument('--allow-running-insecure-content')  # 안전하지 않은 콘텐츠 허용
options.add_argument('--disable-web-security')  # 웹 보안 비활성화

# 드라이버 실행
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 기상청 데이터 페이지로 이동
driver.get("https://data.kma.go.kr/data/grnd/selectAwosRltmList.do?pgmNo=638&tabNo=1")

# 페이지 로드 대기
time.sleep(5)  # 페이지 로드 시간을 보장하기 위해 대기

# 전체 페이지 수 확인
pagination = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CLASS_NAME, "pagination")))
pagination_links = driver.find_elements(By.CSS_SELECTOR, ".pagination > a")
total_pages = int(pagination_links[-2].text)

# 각 페이지에 대해 체크박스 선택 및 다운로드
for page in range(1, total_pages + 1):
    print(f"Processing page {page} of {total_pages}")

    # 체크박스 선택
    checkboxes = WebDriverWait(driver, 20).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "input[type='checkbox']"))
    )
    for checkbox in checkboxes:
        checkbox.click()
    print(f"Page {page}: All checkboxes selected.")

    # 다운로드 버튼 클릭
    download_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.CLASS_NAME, "btn_downfile"))
    )
    download_button.click()
    print(f"Page {page}: Download started.")
    time.sleep(5)  # 다운로드 대기

    # 다음 페이지로 이동
    if page < total_pages:
        next_page = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".pagination > a.next"))
        )
        next_page.click()
        print(f"Page {page}: Moving to next page.")
        time.sleep(5)  # 페이지 로드 대기

print("모든 데이터 다운로드 완료!")
driver.quit()
