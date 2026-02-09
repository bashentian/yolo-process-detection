"""配置验证模块

提供配置验证和类型检查功能，确保配置的正确性和一致性。
"""

from typing import Any, Dict, List, Optional, Type, get_type_hints, get_origin, get_args
from enum import Enum


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


class ConfigValidator:
    """配置验证器

    验证配置类是否符合预期的类型和值范围。
    """

    @staticmethod
    def validate_config(config_class: type, instance: object) -> Dict[str, List[str]]:
        """验证配置实例

        Args:
            config_class: 配置类
            instance: 配置实例

        Returns:
            包含错误信息的字典，格式为 {'field_name': ['error1', 'error2']}
        """
        errors: Dict[str, List[str]] = {}

        # 获取类型提示
        type_hints = get_type_hints(config_class)

        for field_name, expected_type in type_hints.items():
            if not hasattr(instance, field_name):
                errors[field_name] = [f"字段 '{field_name}' 不存在"]
                continue

            value = getattr(instance, field_name)
            field_errors = ConfigValidator._validate_field(
                field_name, value, expected_type
            )
            if field_errors:
                errors[field_name] = field_errors

        return errors

    @staticmethod
    def _validate_field(
        field_name: str, value: Any, expected_type: type
    ) -> List[str]:
        """验证单个字段

        Args:
            field_name: 字段名
            value: 字段值
            expected_type: 期望类型

        Returns:
            错误列表
        """
        errors = []

        # 处理 Optional 类型
        if get_origin(expected_type) is Optional:
            if value is None:
                return []
            expected_type = get_args(expected_type)[0]

        # 处理 List 类型
        if get_origin(expected_type) is list:
            if not isinstance(value, list):
                errors.append(f"期望 list 类型，实际得到 {type(value).__name__}")
            return errors

        # 处理 Dict 类型
        if get_origin(expected_type) is dict:
            if not isinstance(value, dict):
                errors.append(f"期望 dict 类型，实际得到 {type(value).__name__}")
            return errors

        # 处理枚举类型
        if isinstance(expected_type, type) and issubclass(expected_type, Enum):
            if not isinstance(value, expected_type):
                errors.append(
                    f"期望 {expected_type.__name__} 枚举值，实际得到 '{value}'"
                )
            return errors

        # 基本类型检查
        if isinstance(expected_type, type):
            if not isinstance(value, expected_type):
                errors.append(
                    f"期望 {expected_type.__name__} 类型，实际得到 {type(value).__name__}"
                )

        # 范围验证
        if field_name == "CONFIDENCE_THRESHOLD":
            if not 0 <= value <= 1:
                errors.append("置信度阈值必须在 0 到 1 之间")
        elif field_name == "IOU_THRESHOLD":
            if not 0 <= value <= 1:
                errors.append("IOU阈值必须在 0 到 1 之间")
        elif field_name == "MAX_DETECTIONS":
            if value <= 0:
                errors.append("最大检测数必须大于 0")

        return errors

    @staticmethod
    def validate_model_config(config: Any) -> bool:
        """验证模型配置

        Args:
            config: 配置实例

        Returns:
            是否有效

        Raises:
            ConfigValidationError: 配置无效时
        """
        required_fields = [
            "MODEL_NAME",
            "CONFIDENCE_THRESHOLD",
            "IOU_THRESHOLD",
            "MAX_DETECTIONS",
            "DEVICE",
        ]

        for field in required_fields:
            if not hasattr(config, field):
                raise ConfigValidationError(f"缺少必需的配置字段: {field}")

        # 验证 MODEL_NAME
        model_name = getattr(config, "MODEL_NAME", "")
        if not model_name:
            raise ConfigValidationError("MODEL_NAME 不能为空")

        # 验证设备设置
        device = getattr(config, "DEVICE", "cpu")
        if device not in ["cpu", "cuda", "mps"]:
            raise ConfigValidationError(
                f"无效的设备设置: {device}，必须是 'cpu'、'cuda' 或 'mps'"
            )

        return True


def validate_process_config(config: Any) -> bool:
    """验证工序检测配置

    Args:
        config: 配置实例

    Returns:
        是否有效
    """
    validator = ConfigValidator()
    errors = validator.validate_config(type(config), config)

    if errors:
        error_msg = "配置验证失败:\n"
        for field, field_errors in errors.items():
            error_msg += f"  {field}: {', '.join(field_errors)}\n"
        raise ConfigValidationError(error_msg)

    return True


# 快捷验证函数
def validate_detection_config(config: Any) -> bool:
    """验证检测配置"""
    return validate_process_config(config)
