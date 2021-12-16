#!/usr/bin/env python
# -*- coding: utf-8 -*-

from girder.plugin import getPlugin, GirderPlugin
from girder.models.user import User
from .client_webroot import ClientWebroot
from . import rest

# added so the username and password can be set in environment variables
# during the docker build process.  username/password now doesn't have to be
# stored in the code. 
import os
from dotenv import load_dotenv


class ArborNovaGirderPlugin(GirderPlugin):
    DISPLAY_NAME = 'NIH-AIR Adenocarcinoma Inference'

    def _create_anonymous_user(self):

        load_dotenv()
        ANONYMOUS_USER = os.getenv('ANONYMOUS_USER')
        ANONYMOUS_PASSWORD = os.getenv('ANONYMOUS_PASSWORD')

        anon_user = User().findOne({
            'login': ANONYMOUS_USER
        })

        if not anon_user:
            anon_user = User().createUser(
                login=ANONYMOUS_USER,
                password=ANONYMOUS_PASSWORD,
                firstName='Public',
                lastName='User',
                email='anon@example.com',
                admin=False,
                public=False)
            anon_user['status'] = 'enabled'

            anon_user = User().save(anon_user)
        return anon_user

    def load(self, info):
        # Relocate Girder
        info['serverRoot'], info['serverRoot'].girder = (ClientWebroot(),
                                                         info['serverRoot'])
        info['serverRoot'].api = info['serverRoot'].girder.api
        self._create_anonymous_user()
        getPlugin('jobs').load(info)
        info['apiRoot'].arbor_nova = rest.ArborNova()
