# Dự Đoán Giá Máy Tính Xách Tay Bằng Mô Hình Hồi Quy Tuyến Tính

Đây là repository chứa mã nguồn và báo cáo cho dự án **"Dự Đoán Giá Máy Tính Xách Tay Bằng Mô Hình Hồi Quy Tuyến Tính"**, được thực hiện trong học phần Phân Tích Dữ Liệu tại Trường Đại học Công Nghiệp Hà Nội.

## Mục Lục
- [Giới Thiệu](#giới-thiệu)
- [Nguồn Dữ Liệu](#nguồn-dữ-liệu)
- [Phương Pháp và Công Cụ](#phương-pháp-và-công-cụ)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
- [Lời Cảm Ơn](#lời-cảm-ơn)

## Giới Thiệu
Mục tiêu của dự án là phân tích và dự đoán giá máy tính xách tay dựa trên các thông số kỹ thuật của chúng bằng cách áp dụng mô hình hồi quy tuyến tính. Dự án nhằm giúp hiểu rõ các yếu tố ảnh hưởng đến giá sản phẩm và cung cấp những thông tin hữu ích cho người tiêu dùng và doanh nghiệp.

## Nguồn Dữ Liệu
Dữ liệu được sử dụng trong dự án này được thu thập, sửa đổi và trích xuất từ [Amazon Laptop Analysis của Milad Nooraei](https://github.com/MiladNooraei/Amazon-Laptop-Analysis). Bộ dữ liệu bao gồm các thông số chi tiết về máy tính xách tay cùng với giá bán - là biến mục tiêu.

## Phương Pháp và Công Cụ
- **Ngôn ngữ lập trình**: Python
- **Thư viện chính**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **Công cụ phát triển**: PyCharm

Phương pháp chính được áp dụng là hồi quy tuyến tính để phân tích mối quan hệ giữa các thông số kỹ thuật và giá bán.

## Hướng Dẫn Sử Dụng
1. Tải về hoặc clone repository này:
   ```bash
   git clone https://github.com/yourusername/your-repo.git

2. Cài đặt các thư viện cần thiết bằng lệnh sau:
   ```bash
   pip install -r requirements.txt
3. Chạy mã nguồn chính hoặc các tập tin liên quan để thực hiện dự đoán:
   ```bash
   python laptop_linear_regression.py

## Lời Cảm Ơn

Dự án này được thực hiện dưới sự hướng dẫn của thầy **TS. Nguyễn Mạnh Cường**.  
Chúng tôi xin chân thành cảm ơn thầy vì những đóng góp và chỉ dẫn quý báu trong suốt quá trình thực hiện.

Dữ liệu và ý tưởng ban đầu được lấy cảm hứng từ dự án của [Milad Nooraei](https://github.com/MiladNooraei/Amazon-Laptop-Analysis).  
Chúng tôi đã thực hiện sửa đổi và bổ sung để phù hợp với mục tiêu của dự án.

### Nhóm thực hiện:
- Nguyễn Trung Hiếu  
- Vương Trí Tín  
- Trần Văn Trường  
