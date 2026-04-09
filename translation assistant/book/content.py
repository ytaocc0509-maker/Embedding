from enum import Enum, auto
import re
import pandas as pd
from utils.log_utils import log


class ContentType(Enum):
    """ 内容的类型 枚举"""
    TEXT = auto()
    TABLE = auto()
    IMAGE = auto()


class Content:
    """ 书中的文本内容"""

    def __init__(self, content_type: ContentType, original, translation=None):
        """
        内容的初始化
        :param content_type: 类型
        :param original: 原文
        :param translation: 翻译之后
        """
        self.content_type = content_type
        self.original = original
        self.translation = translation
        self.status = False  # 翻译完成的状态

    def set_translation(self, translation, status):
        """设置翻译之后的文本，并设置翻译状态"""
        if self.content_type == ContentType.TEXT and isinstance(translation, str) and status:
            self.translation = translation
            self.status = status
        else:
            log.warning('当前翻译之后的文本内容不是字符串，请检查！')

    def get_original_to_string(self):
        """把对象变成一个字符串"""
        return self.original


class TableContent:
    """ 书中的表格内容"""

    def __init__(self, content_type: ContentType, original, translation=None):
        """
        内容的初始化
        :param content_type: 类型
        :param original: 原文
        :param translation: 翻译之后
        """
        df = pd.DataFrame(original)
        self.content_type = content_type
        self.original = df
        self.translation = translation
        self.status = False  # 翻译完成的状态

    @log.catch
    def set_translation(self, translation, status):
        """
        设置翻译之后的文本，并设置翻译状态

        """


    def get_original_to_string(self):
        """把dataframe对象变成一个字符串"""
        return self.original.to_string(header=False, index=False)
