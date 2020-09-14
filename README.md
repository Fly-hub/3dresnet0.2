# 3dresnet0.2
0.2
本项目使用UCF-101的trainlist1数据划分进行训练和测试。需要在运行前对数据进行一些预处理，预处理的步骤如下：  
- 执行脚本avi2jpg.py，将UCF-101数据中的视频文件逐帧处理为jpg文件并保存在以视频名称命名的文件夹下  
- 执行脚本jpg2pkl.py，将同一视频对应的jpg文件的路径以及标签保存在以视频命名的pkl文件中。
- 执行脚本data_list_gener.py，生成对应的train.list，test.list文件
