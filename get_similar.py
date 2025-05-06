import requests

port = 8080
image_path = r"C:\Users\Vasiliy\Desktop\flowers\daisy\5547758_eea9edfd54_n.jpg"
url = 'http://localhost:{}/predict'.format(port)
files = {'img_file': open(image_path, 'rb')}
response = requests.post(url, files=files)
print(response.content)

