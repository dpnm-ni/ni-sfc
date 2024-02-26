# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from server.models.base_model_ import Model
from server import util


class SFCInfo(Model):
    def __init__(self, sfcr_name: str=None, sfc_name: str=None):  # noqa: E501

        self.swagger_types = {
            'sfcr_name': str,
            'sfc_name': str
        }

        self.attribute_map = {
            'sfcr_name': 'sfcr_name',
            'sfc_name': 'sfc_name'
        }

        self._sfcr_name = sfcr_name
        self._sfc_name = sfc_name

    @classmethod
    def from_dict(cls, dikt) -> 'SFCInfo':
        return util.deserialize_model(dikt, cls)

    @property
    def sfcr_name(self) -> str:
        return self._sfcr_name

    @sfcr_name.setter
    def sfcr_name(self, sfcr_name: str):
        self._sfcr_name = sfcr_name

    @property
    def sfc_name(self) -> str:
        return self._sfc_name

    @sfc_name.setter
    def sfc_name(self, sfc_name: str):
        self._sfc_name = sfc_name


class SFCInfo_DQN(Model):
    def __init__(self, sfcr_name: str=None, sfc_name: str=None, training_mode: bool=False):  # noqa: E501

        self.swagger_types = {
            'sfcr_name': str,
            'sfc_name': str,
            'training_mode': bool
        }

        self.attribute_map = {
            'sfcr_name': 'sfcr_name',
            'sfc_name': 'sfc_name',
            'training_mode': 'training_mode'
        }

        self._sfcr_name = sfcr_name
        self._sfc_name = sfc_name
        self._sfc_boolean = training

    @classmethod
    def from_dict(cls, dikt) -> 'SFCInfo':
        return util.deserialize_model(dikt, cls)

    @property
    def sfcr_name(self) -> str:
        return self._sfcr_name

    @sfcr_name.setter
    def sfcr_name(self, sfcr_name: str):
        self._sfcr_name = sfcr_name

    @property
    def sfc_name(self) -> str:
        return self._sfc_name

    @sfc_name.setter
    def sfc_name(self, sfc_name: str):
        self._sfc_name = sfc_name

    @property
    def training_mode(self) -> bool:
        return self._training_mode

    @sfcr_name.setter
    def sfcr_name(self, sfcr_name: str):
        self._training_mode = training_mode
