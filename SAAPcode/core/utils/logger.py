import os
import sys
import time
import codecs
import logging


class LoggerWithDepth():
    def __init__(self, env_name, config, root_dir = 'runtime_log', overwrite = True, setup_sublogger = True):
        if os.path.exists(os.path.join(root_dir, env_name)) and not overwrite:
            raise Exception("Logging Directory {} Has Already Exists. Change to another name or set OVERWRITE to True".format(os.path.join(root_dir, env_name)))
        
        self.env_name = env_name
        self.root_dir = root_dir
        self.log_dir = os.path.join(root_dir, env_name)
        self.overwrite = overwrite

        self.format = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                        "%Y-%m-%d %H:%M:%S")

        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        
        self.best_checkpoint_path = os.path.join(self.log_dir, 'pytorch_model.bin')

        if setup_sublogger:
            sub_name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
            self.setup_sublogger(sub_name, config)

    def setup_sublogger(self, sub_name, sub_config):
        self.sub_dir = os.path.join(self.log_dir, sub_name)
        if os.path.exists(self.sub_dir):
            raise Exception("Logging Directory {} Has Already Exists. Change to another sub name or set OVERWRITE to True".format(self.sub_dir))
        else:
            os.mkdir(self.sub_dir)

        with open(os.path.join(self.sub_dir, 'train.sh'), 'w') as f:
            f.write('python ' + ' '.join(sys.argv))

        log_format=logging.Formatter("%(asctime)s - %(levelname)s :       %(message)s", "%Y-%m-%d %H:%M:%S")
        
        logger_name = f"saap.{self.env_name}.{sub_name}"
        self.writer = logging.getLogger(logger_name)
        self.writer.handlers.clear()
        self.writer.propagate = False

        fileHandler = logging.FileHandler(os.path.join(self.sub_dir, "training.log"))
        fileHandler.setFormatter(log_format)
        self.writer.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(log_format)
        self.writer.addHandler(consoleHandler)

        self.writer.setLevel(logging.INFO)

        self.checkpoint_path = os.path.join(self.sub_dir, 'pytorch_model.bin')      
        self.lastest_checkpoint_path = os.path.join(self.sub_dir, 'latest_model.bin')   

    def log(self, info):
        text = '' if info is None else str(info)
        suppressed_prefixes = (
            '[distill_rerank_v2]',
            '[distill_rerank_v3]',
            '[distill_rerank_v4]',
            '[post_taylor_swap]',
            '[attn_post_taylor_swap]',
        )
        if text.startswith(suppressed_prefixes):
            return
        self.writer.info(info)
        for handler in self.writer.handlers:
            try:
                handler.flush()
            except Exception:
                pass

    def write_description_to_folder(self, file_name, config):
        with codecs.open(file_name, 'w') as desc_f:
            desc_f.write("- Training Parameters: \n")
            for key, value in config.items():
                desc_f.write("  - {}: {}\n".format(key, value))
