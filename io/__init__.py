"""
Provides Adapters to read binary files generated from the following equipment:

* **LeCroy oscilloscopes** : Provided by :func:`utkit.io.lecroy.lecroy`
* **SAFT scanners** : Provided by :func:`utkit.io.saft.saft`


.. autofunction:: utkit.io.lecroy.lecroy

.. autofunction:: utkit.io.saft.saft

"""
# from __future__ import absolute_import

# import io.saft
# import io.lecroy
from .saft import saft
from .lecroy import lecroy
