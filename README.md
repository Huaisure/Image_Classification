# Image_Classification

- 训练模型，在当前目录下
    ```bash
    python train.py
    ```

- 评估模型
    ```bash
    python evaluate.py
    ```
    注意：模型的路径需要在evaluate.py中提前修改

- 观察热力图
    ```bash
    python grad_cam.py
    ```
    需要输入对应的图片序号


- 需要删除某几类时进行训练
    ```bash
    python train_less_category.py
    ```
    请在 `train_less_category.py` 中修改对应要删除的序号