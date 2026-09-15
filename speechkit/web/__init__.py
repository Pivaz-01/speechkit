"""
The local web interface.

Deliberately does not re-export the Flask instance. The module is
``speechkit.web.app`` and the Flask object inside it is also called ``app``, so
re-exporting it here would shadow the submodule and make
``from speechkit.web import app`` hand back the Flask object instead of the
module. Import the factory, or the WSGI target ``speechkit.web.app:app``.
"""

from .app import create_app, serve

__all__ = ["create_app", "serve"]
