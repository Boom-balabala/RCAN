from importlib import import_module

from dataloader import MSDataLoader
from torch.utils.data.dataloader import default_collate

class Data:
    def __init__(self, args):
        kwargs = {}
        if not args.cpu:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False
        else:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False

        self.loader_train = None
        # args.test_only:'set this option to test the model'
        if not args.test_only:
            # .lower()转换字符串中所有大写字符为小写
            # args.data_train:'train dataset name'-default:'DIV2K' ：指定了训练集
            module_train = import_module('data.' + args.data_train.lower())
            # trainset='train dataset name'
            trainset = getattr(module_train, args.data_train)(args)
            self.loader_train = MSDataLoader(
                args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                **kwargs #这个是上面的那个字典
            )

        if args.data_test in ['Set5', 'Set14', 'B100', 'Urban100']:
            # benchmark_noise ？
            if not args.benchmark_noise:
                module_test = import_module('data.benchmark')
                testset = getattr(module_test, 'Benchmark')(args, train=False)
            else:
                module_test = import_module('data.benchmark_noise')
                testset = getattr(module_test, 'BenchmarkNoise')(
                    args,
                    train=False
                )
        # 设置模型测试集
        else:
            # data.div2k，然后再走一遍和训练集一样的流程
            module_test = import_module('data.' +  args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, train=False)

        self.loader_test = MSDataLoader(
            args,
            testset,
            batch_size=1,
            shuffle=False,
            **kwargs
        )
