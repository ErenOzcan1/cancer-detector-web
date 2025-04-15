import requests

# URL for the predict endpoint
url = 'http://127.0.0.1:5000/'

# Replace 'your_test_image.jpg' with the path to a test image
with open('your_test_image.jpg', 'rb') as img_file:
    files = {'image': img_file}
    response = requests.post(url, files=files)

# Save the response image as 'result.jpg'
with open('result.jpg', 'wb') as out_file:
    out_file.write(response.content)

print("Result image saved as result.jpg")
