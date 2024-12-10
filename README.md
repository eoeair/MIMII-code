# MIMII-code

有监督学习部分已经全部写完，无监督部分正在进行中

对于这种任务，思路是：先对snr做无监督学习，因为不同snr下模型表现可能不同，然后根据无监督模型的结果，再对label/device做分类

也就是说，这是一个两阶段模型，包括模态识别与单模态下的分类

TODO：全流程，无监督（事实上，数据本身就是要求无监督）

## Data pre

see my [Pre_data](https://github.com/ben0i0d/Pre_Data)

copy to `data`

## RUN

see `process.ipynb`