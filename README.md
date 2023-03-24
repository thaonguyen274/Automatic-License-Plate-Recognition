
# License-Plate-Recognition

Thuật toán nhận dạng biển số xe bao gồm 3 bước:

* Xác định vùng chứa biển số xe (Sử dụng YOLOv8)
* Xác định biển số là biển 1 dòng hay 2 dòng.
* Sử dụng module text recognition (Sử dụng backbone RCNN và CTC) để nhận dạng biển số xe.

Nhược điểm:
* Khi ảnh đầu vào bị đặt một góc quá nghiêng thì một vài kí tự sẽ bị nhầm dòng. Có một cách giải quyết là dùng một mạng transformer xoay ảnh nghiêng về ảnh thẳng.
*  Hoạt động kém khi bức ảnh quá mờ, lóa, chói.

## DATASET

Dataset là tập dữ liệu công khai được thu thập trên Roboflow, gồm 2000 ảnh xe máy, ô tô đi qua khu vực gửi xe trong bãi đỗ xe. Ảnh được ghi lại trong nhiều điều kiện khác nhau: ánh sáng tốt, thiếu sáng, bị đèn chiếu, lóa,... 

```
!pip install roboflow --quiet

from roboflow import Roboflow
rf = Roboflow(api_key="Tno3ltYs1pK7NVjkI3Ke")
project = rf.workspace("license-plate-detect").project("new_biensoxe")
dataset = project.version(3).download("yolov8")

```

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
├── test_plate.py 
├── train.ipynb

```

* modules: thư mục chứa các định nghĩa module con của thuật toán text recognition.
* best_accuracy.pth: file mô hình thuật toán text recognition.
* best.pt: file mô hình thuật toán phát hiện biển số xe.
* model.py: file định nghĩa thuật toán text recognition.
* ocrmodule.py: file chứa code inference thuật toán text recognition.
* test_plate.py: file main chính chạy end-to-end inference của project. 
* train.ipynb: file notebook dùng để train mô hình YOLOv8, lấy được file weight của mô hình. 

## Install Enviroments 

```
pip install -r requirements.txt
``` 

##  Chạy thuật toán
Có thể chạy ngay bằng câu lệnh, trước đó cần thay thế path file ảnh muốn đọc trong file 

```
python test_plate.py
```
## Kết quả 

