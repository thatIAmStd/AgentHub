

def save_img(location,img_bytes):
    try:
        output_file = location  # 可以根据需要修改文件名或路径
        with open(output_file, "wb") as f:
            f.write(img_bytes)
    except Exception as e:
        print(f"Error generating graph: {e}")