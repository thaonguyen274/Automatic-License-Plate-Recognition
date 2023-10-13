
# License-Plate-Recognition

Thuật toán nhận dạng biển số xe bao gồm 3 bước:

* Xác định vùng chứa biển số xe (Sử dụng YOLOv8)
* Xác định biển số là biển 1 dòng hay 2 dòng.
* Sử dụng module text recognition (Sử dụng backbone RCNN và CTC) để nhận dạng biển số xe.

Nhược điểm:
* Khi ảnh đầu vào bị đặt một góc quá nghiêng thì một vài kí tự sẽ bị nhầm dòng. Có một cách giải quyết là dùng một mạng transformer xoay ảnh nghiêng về ảnh thẳng.
*  Hoạt động kém khi bức ảnh quá mờ, lóa, chói.

## DATASET

Dataset là tập dữ liệu công khai được thu thập trên Roboflow, gồm ảnh xe máy, ô tô đi qua khu vực gửi xe trong bãi đỗ xe. Ảnh được ghi lại trong nhiều điều kiện khác nhau: ánh sáng tốt, thiếu sáng, bị đèn chiếu, lóa,... Bên cạnh đó, hình ảnh xe ô tô đi qua trạm thu phí cũng được thu thạp thêm để tăng sự đa dạng cho dữ liệu. 


Cấu trúc dataset:
```
├──test  ├── images
         ├── labels 
├──train ├── images
         ├── labels 
├──valid ├── images
         ├── labels 
├── data.yaml 
├── README 
```

## Cấu trúc code
```
├── modules 
├── best_accuracy.pth 
├── best.pt 
├── model.py 
├── ocrmodule.py 
├── predict_plate.py 
├── cer.ipynb
├── MODEL
├── ground_truth 
├── video 



```

* modules: thư mục chứa các định nghĩa module con của thuật toán text recognition.
* best_accuracy.pth: file mô hình thuật toán text recognition.
* best.pt: file mô hình thuật toán phát hiện biển số xe.
* model.py: file định nghĩa thuật toán text recognition.
* ocrmodule.py: file chứa code inference thuật toán text recognition.
* predict_plate.py: file main chính chạy end-to-end inference của project. 
* cer.ipynb: file tính và in kết qủa CER metrics của 100 ảnh lấy từ tập ``` ground_truth``` 
* Model: thư mục chứa code training mô hình YOLOv8, kết quả training được lưu trong file ```ultralytics/runs```
* ground_truth: Ảnh biển số xe và file txt đính kèm thông tin xuất hiện trong biển 
* video: dùng để test cho phần ``` predict_plate.py```

## Install Enviroments 

```
pip install -r requirements.txt
``` 

##  Chạy thuật toán
Có thể chạy ngay bằng câu lệnh, trước đó cần thay thế path file video muốn đọc trong file 

```
python predict_plate.py
```
## Kết quả 

![alt-text](/home/ubuntu/Documents/ALPR/video_1.gif)
![alt-text](/home/ubuntu/Documents/ALPR/video.gif)
