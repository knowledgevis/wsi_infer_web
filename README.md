
==========
web_apps_container
==========

A web apps plugin for the python, opensource storage and task manager, Girder 3.

This application uses Girder 3 to serve a webpage which can invoke
python functions through AJAX calls to perform operations.  An instance
of girder_worker services the API to run jobs invoked through the web interface.

To add a new mini-application to the web interface:

Add python file for app function in ttt/app_support
* Edit __init__.py in ttt/app_support
* Edit rest.py in girder_plugin/arbor_nova with import statement, self.route statement, and a definition wrapping the function and parameters
* Add vue definition in client/src/apps
* Add picture for app in client/src/assets
* Add app to the home page by editing Home.vue in client/src/views
* Add router to app in router.js in client/src
* Run pip install -e . in girder_plugin
* Run pip install -e . in girder_worker_tasks
* Run yarn build in /client, rm the old dist and cp the new one
* Restart girder_worker.service and girder.service
    (sudo systemctl restart girder, sudo systemctl restart girder_worker)

For debugging purposes, it is easier to run 'girder serve' and girder_worker by hand. 
To run girder_worker, do "python -m girder_worker" in a shell with the correct environment



Installation
------------

This is descriptive rather than prescriptive, but it is what has been tested.

* Do this work with Python3
* Have a virtualenv or use conda
* Install mongo and rabbitmq

* In virtualenv **girder** run the following commands, it doesn't matter where you run them from:

.. code-block:: bash

    $ pip install --pre girder[plugins]
    $ girder build

* These commands need to be run in the **girder** virtualenv from specific locations.

.. code-block:: bash

    $ cd wsi_infer_web/girder_worker_tasks    
    $ pip install -e .                     # install gw tasks for producer
    $ cd ../../wsi_infer_web/girder_plugin
    $ pip install -e .                     # install girder plugin
    $ girder serve                         # start serving girder
 

* In virtualenv **girder** run the following command, it doesn't matter where you run it from:

.. code-block:: bash

    $ pip install --pre girder-worker

* These commands need to be run in the **girder** virtualenv from a specific location

.. code-block:: bash

    $ cd wsi_infer_web/girder_worker_tasks    
    $ pip install -e .                     # install gw tasks for consumer
    $ girder-worker                        # start girder-worker




TODO
----

* uploaded files are stored in girder permanently  Is there a way to clean up?
