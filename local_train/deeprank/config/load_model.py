import logging
import os

import tensorflow as tf
from tensorflow.contrib.framework import filter_variables
#from utils.registry import registry


logger = logging.getLogger(__name__)


#@registry.register_hook("restore_hook")
class RestoreCheckpointHook(tf.train.SessionRunHook):
    def __init__(
        self,
        restore_checkpoint_dir,
        init_global_step=True,
        exclude_patterns=None,
        include_patterns=None,
    ):
        self.restore_checkpoint_dir = restore_checkpoint_dir
        self.exclude_patterns = exclude_patterns if exclude_patterns is not None else []
        self.include_patterns = include_patterns

        if init_global_step:
            self.exclude_patterns.append("global_step")

    def begin(self):
        if tf.io.gfile.exists(self.restore_checkpoint_dir + "/checkpoint"):
            variables_to_restore = filter_variables(
                tf.global_variables(),
                include_patterns=self.include_patterns,
                exclude_patterns=self.exclude_patterns,
                reg_search=True
            )
            self.saver = tf.train.Saver(variables_to_restore)
            logger.info("exclude_patterns: {0}".format(self.exclude_patterns))
            logger.info("include_patterns: {0}".format(self.include_patterns))
            logger.info("variables_to_restore: {0}".format(variables_to_restore))

    def after_create_session(self, session, coord):
        ckpt = tf.train.get_checkpoint_state(self.restore_checkpoint_dir)
        if ckpt is not None:
            ckpt = ckpt.model_checkpoint_path
            logger.info("Restore checkpoint from {}".format(ckpt))
            self.saver.restore(session, ckpt)
        else:
            logger.info("No checkpoint in {}, model train from init".format(self.restore_checkpoint_dir))
    