import os
import logging
##################################################
##### 路径和文件名
#总目录
PATH_PROJECT = os.path.dirname(os.path.abspath(__file__))
#攻击方法目录
PATH_ATTACK_METHOD = PATH_PROJECT + "Attack/"
#防御方法目录
PATH_DEFENSE_METHOD = PATH_PROJECT + "Defense/"
#评估方法目录
PATH_EVALUATION_METHOD = PATH_PROJECT + "Evaluation/"
#日志目录
PATH_LOG = PATH_PROJECT + "Log/"
#模型存放目录
PATH_MODEL = PATH_PROJECT + "Model/"
#配置存放目录
PATH_CONFIG_RELATIVE = "Config/" # 要注意配置是相对路径，需要加上上方对应的目录才会生效
#模型检查点目录
PATH_MODEL_CHECKPOINT = PATH_MODEL + "Checkpoint/"
#模型类型定义存放目录
PATH_MODEL_TYPE = PATH_MODEL + "ModelType/"
#用户模型定义存放目录
PATH_USER_MODEL = PATH_MODEL + "UserModel/"
#默认日志文件名
FILENAME_DEFAULT_LOG="DefaultLog.txt"
#包名
MODEL_ATTACK="Attack."
MODEL_DEFENSE="Defense."
MODEL_EVALUATION="Evaluation."
MODEL_MODEL="Model."
MODEL_MODEL_TYPE="Model.ModelType."
MODEL_DATA_SET="DataSet."
##################################################

##################################################
##### 参数
#日志打印格式
LOGGING_FORMAT='%(asctime)s|%(name)s|%(levelname)s[%(filename)s-%(funcName)s]:%(message)s'
LOGGING_LEVEL_DEBUG=logging.DEBUG
LOGGING_LEVEL_INFO=logging.INFO
LOGGING_LEVEL_WARNING=logging.WARNING
LOGGING_LEVEL_ERROR=logging.ERROR
LOGGING_LEVEL_CRITICAL=logging.CRITICAL
##################################################

##################################################
##### 其他

##################################################