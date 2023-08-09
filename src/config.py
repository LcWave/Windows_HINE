import configparser
import os


class Config(object):
    def __init__(self, config_file, args):
        conf = configparser.ConfigParser()
        data_path = os.getcwd()
        try:
            conf.read(config_file)
        except:
            print("failed!")

        self.num_walks = conf.getint("common_para", "num_walks")
        self.walk_length = conf.getint("common_para", "walk_length")
        self.window_size = conf.getint("common_para", "window_size")
        self.neg_num = conf.getint("common_para", "neg_num")
        self.batch_size = conf.getint("common_para", "batch_size")
        self.dim = conf.getint("common_para", "dim")
        self.num_workers = conf.getint("common_para", "num_workers")

        self.alpha = conf.getfloat("common_para", "alpha")
        self.epochs = conf.getint("common_para", "epochs")
        self.seed = conf.getint("common_para", "seed")
        self.lr_decay = conf.getfloat("common_para", "lr_decay")
        self.log_dir = conf.get("common_para", "log_dir")
        self.log_interval = conf.getint("common_para", "log_interval")



        # data_setup
        self.data_type = conf.get(args.dataset, "data_type")
        self.relation_list = conf.get(args.dataset, "relation_list")


        # training dataset path
        self.output_modelfold = conf.get("Data_Out", "output_modelfold")
        self.input_fold = conf.get("Data_In", "input_fold") + args.dataset  + '/'
        self.out_emd_file = conf.get("Data_Out", "out_emd_file") + args.model + "/"
        self.temp_file = conf.get("Data_Out", "temp_file") + args.model + "/"


        # ____________dataset________________
        self.node_size = conf.getint(args.dataset, "node_size")
        self.link_size = conf.getint(args.dataset, "link_size")




        # ____________________model____________________


        if args.model == "RHINE":
            self.relation_category = conf.get("RHINE", "relation_category")

            self.data_set = conf.get("Model_Setup", "data_set")
            self.combination = conf.get("RHINE", "combination")
            self.link_type = conf.get("RHINE", "link_type")
            self.mode = conf.get("Model_Setup", "mode")
            self.IRs_nbatches = conf.getint("RHINE", "IRs_nbatches")
            self.ARs_nbatches = conf.getint("RHINE", "ARs_nbatches")

            self.margin = conf.getint("RHINE", "margin")
            self.ent_neg_rate = conf.getint("Model_Setup", "ent_neg_rate")
            self.rel_neg_rate = conf.getint("Model_Setup", "rel_neg_rate")
            self.evaluation_flag = conf.get("Model_Setup", "evaluation_flag")
            self.log_on = conf.getint("Model_Setup", "log_on")
            self.exportName = conf.get("Model_Setup", "exportName")
            if self.exportName == 'None':
                self.importName = None
            self.importName = conf.get("Model_Setup", "importName")
            if self.importName == 'None':
                self.importName = None
            self.export_steps = conf.getint("Model_Setup", "export_steps")
            self.opt_method = conf.get("Model_Setup", "opt_method")
            self.optimizer = conf.get("Model_Setup", "optimizer")
            if self.optimizer == 'None':
                self.optimizer = None
            self.weight_decay = conf.get("Model_Setup", "weight_decay")
        elif args.model == "HERec":
            self.metapath_list = conf.get("HERec", "metapath_list")
        elif args.model == "Metapath2vec":
            self.num_walks = conf.getint("Metapath2vec", "num_walks")
            self.walk_length = conf.getint("Metapath2vec", "walk_length")
            self.window_size = conf.getint("Metapath2vec", "window_size")
            self.neg_num = conf.getint("Metapath2vec", "neg_num")
            self.batch_size = conf.getint("Metapath2vec", "batch_size")
            self.dim = conf.getint("Metapath2vec", "dim")
            self.num_workers = conf.getint("Metapath2vec", "num_workers")
            self.alpha = conf.getfloat("Metapath2vec", "alpha")
            self.epochs = conf.getint("Metapath2vec", "epochs")
            self.metapath = conf.get("Metapath2vec", "metapath")

        elif args.model == "HeteSpaceyWalk":
            self.metapath = conf.get("HeteSpaceyWalk", "metapath")
            self.beta = conf.getfloat("HeteSpaceyWalk", "beta")
        elif args.model == "DHNE":
            self.scale = conf.get("DHNE", "scale")
            self.hidden_size = conf.getint("DHNE", "hidden_size")
            self.prefix_path = conf.get("DHNE", "prefix_path")
            self.triple_hyper = conf.get("DHNE", "triple_hyper")
        elif args.model == "HHNE":
            self.metapath = conf.get("HHNE", "metapath")
        elif args.model == "MetaGraph2vec":
            self.care_type = conf.getint("MetaGraph2vec", "care_type")
            self.max_keep_model = conf.getint("MetaGraph2vec", "max_keep_model")
        elif args.model == "PME":
            self.dimensionR = conf.getint("PME", "dimensionR")
            self.no_validate = conf.getint("PME", "no_validate")
            self.margin = conf.getint("PME", "margin")
            self.nbatches = conf.getint("PME", "nbatches")
            self.loadBinaryFlag = conf.getint("PME", "loadBinaryFlag")
            self.outBinaryFlag = conf.getint("PME", "outBinaryFlag")
            self.M = conf.getint("PME", "M")
        elif args.model == "HAN":
            self.dim = conf.getint("HAN", "dim")
            self.alpha = conf.getfloat("HAN", "alpha")
            self.epochs = conf.getint("HAN", "epochs")
            self.lr_decay = conf.getfloat("HAN", "lr_decay")
            self.patience = conf.getint("HAN", "patience")
            self.mp_list = conf.get("HAN", "metapath_list")
            self.featype = conf.get("HAN", "featype")
        elif args.model == "HeGAN":
            self.lambda_gen = conf.getfloat("HeGAN", "lambda_gen")
            self.lambda_dis = conf.getfloat("HeGAN", "lambda_dis")
            self.n_sample = conf.getint("HeGAN", "n_sample")
            self.lr_gen = conf.getfloat("HeGAN", "lr_gen")
            self.lr_dis = conf.getfloat("HeGAN", "lr_dis")
            self.n_epoch = conf.getint("HeGAN", "n_epoch")
            self.saves_step = conf.getint("HeGAN", "saves_step")
            self.sig = conf.getfloat("HeGAN", "sig")
            self.d_epoch = conf.getint("HeGAN", "d_epoch")
            self.g_epoch = conf.getint("HeGAN", "g_epoch")
            self.n_emb = conf.getint("HeGAN", "n_emb")
            self.pretrain_node_emb_filename = conf.get("HeGAN", "pretrain_node_emb_filename")
            self.emb_filenames = self.out_emd_file
            self.model_log = self.output_modelfold + 'HeGAN/'
            self.label_smooth = conf.getfloat("HeGAN", "label_smooth")
        elif args.model == "PTE":
            self.iteration = conf.getint("PTE", "iteration")
        elif args.model == "CKD":
            self.dataset = args.dataset
            self.model = args.model
            self.target_node = conf.get("CKD","target_node")
            self.transform = conf.getboolean("CKD","transform")
            self.attributed = conf.getboolean("CKD","attributed")
            self.supervised = conf.getboolean("CKD","supervised")
            self.version = conf.get("CKD","version")
            self.ltype = conf.get("CKD","ltype")
            self.seed = conf.getint("CKD","seed")
            self.device = conf.get("CKD","device")
            self.size = conf.getint("CKD","size")
            self.layers = conf.getint("CKD","layers")
            self.dropout = conf.getfloat("CKD","dropout")
            self.negative_cnt = conf.getint("CKD","negative_cnt")
            self.sample_times = conf.getint("CKD","sample_times")
            self.topk = conf.getint("CKD","topk")
            self.neigh_por = conf.getfloat("CKD","neigh_por")
            self.lr = conf.getfloat("CKD","lr")
            self.batch_size = conf.getint("CKD","batch_size")
            self.epochs = conf.getint("CKD","epochs")
            self.stop_cnt = conf.getint("CKD","stop_cnt")
            self.global_weight = conf.getfloat("CKD","global_weight")
        elif args.model == "AspEm":
            self.dataset = args.dataset
            self.model = args.model
            self.target_node = conf.get("AspEm", "target_node")
            self.binary = conf.getboolean("AspEm", "binary")
            self.size = conf.getint("AspEm", "size")
            self.negative = conf.getint("AspEm", "negative")
            self.samples = conf.getfloat("AspEm", "samples")
            self.alpha = conf.getfloat("AspEm", "alpha")
            self.threads = conf.getint("AspEm", "threads")
        else:
            pass
