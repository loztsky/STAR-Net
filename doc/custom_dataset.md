### 1.自制数据集
#### 1.1 CarradaDataset(卡拉达数据集)
数据集目录层级
```txt
- custom_dir
    - data_seq_ref.json
    - light_dataset_frame_oriented.json
    - seq_name1
        - range_doppler_processed
            - 000000.npy
            - 000001.npy
            - ...
        - range_doppler_raw (文件夹内同range_doppler_processed)
        - range_angle_processed (文件夹内同range_doppler_processed)
        - range_angle_raw (文件夹内同range_doppler_processed)
        - angle_doppler_processed (文件夹内同range_doppler_processed)
        - angle_doppler_raw (文件夹内同range_doppler_processed)
        - annotations
            - box
            - dense
                - 000000
                    - range_angle.npy
                    - range_doppler.npy
                - 000001
                - ...
            - sparse
    - seq_name2
    - seq_name3
    - xxx
```
以上面的数据为例，自制数据集目录层级如下
* custom_dir 自制数据集目录路径
    * data_seq_ref 数据集相关信息
    * light_dataset_frame_oriented 数据集信息
    * seq_name1 数据的第一个序列
        * range_doppler_processed ：rd图 (npy文件夹)
        * range_doppler_raw ： rd图原生数据，tp图
        * range_angle_processed
        * range_angle_raw
        * angle_doppler_processed
        * angle_doppler_raw
        * annotations
            * box 目标框信息文件夹
            * dense 语义分割
            * sparse 实例分割
    * seq_name2 数据的第一个序列
        * range_doppler_processed ：rd图 (npy文件夹)
        * range_doppler_raw ： rd图原生数据，tp图
        * range_angle_processed
        * range_angle_raw
        * angle_doppler_processed
        * angle_doppler_raw
        * annotations
            * box 目标框信息文件夹
            * dense 语义分割
            * sparse 实例分割
    * ...

注意的是，六种数据都必须要有，否则内部读取会报错。

#### 1.2 OTHR dataset (天波超视距雷达数据集)
这是我自己设计的数据集,用于适配OTH雷达的多时序RDM数据。目录层级如下
* custom_dir 自制数据集目录路径
    * train
        * y-m-d-h-m-s (年-月-日-时-分-秒)
            * range_doppler_processed ：rd图 (npy文件夹)
                * 000000.npy
                * 000001.npy
                * ...
            * range_doppler_raw ： rd图原生数据，tp图
                * 000000.npy
                * 000001.npy
                * ...
            * annotations
                * 000000
                    * range_doppler.npy
                    * range_angle.npy
                * 000001
                    * range_doppler.npy
                    * range_angle.npy
                * ...

    * val
        * y-m-d-h-m-s (年-月-日-时-分-秒)
            * range_doppler_processed ：rd图 (npy文件夹)
            * range_doppler_raw ： rd图原生数据，tp图
            * annotations
    * test
        * y-m-d-h-m-s (年-月-日-时-分-秒)
            * range_doppler_processed ：rd图 (npy文件夹)
            * range_doppler_raw ： rd图原生数据，tp图
            * annotations




### 2.相关配置文件修改
#### 2.1 时序多帧
对于OTH雷达，修改类别数目
* nb_classes
* nb_input_channels
* w_size
* h_size

添加的config选项
* resume 是否继续训练
* val_epoch 验证轮数
* schedular 学习率调整器，仅支持"exp"和"coswarm"
* n_Input 
    * 1: rd
    * 2: rd , ra
    * 3: rd , ra, ad
* n_Output 
    * 1: rd_mask
    * 2: rd_mask , ra_mask

(n_Input,n_Outout)目前仅仅支持(1,1)，(2,2), (3,2).数据集读取，对于模型，
* Input 
    * 1: rd = [B,1,n_frames,H,W]
    * 2: rd , ra = [B,1,n_frames,H,W] ,
    * 3: rd , ra, ad
* Output 
    * 1: rd
    * 2: rd , ra

### 3.附录
#### 3.1数据集相关的类
按照调用属性的类
* Paths
* Carrada
* SequenceCarradaDataset
* CarradaDataset

#### 3.2 数据集相关的文件
* model_config
* mvrss/config_files/config.ini
* self.carrada / 'data_seq_ref.json'
* self.carrada / 'light_dataset_frame_oriented.json'
