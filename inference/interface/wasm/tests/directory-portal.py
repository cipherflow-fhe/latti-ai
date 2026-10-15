"""Test-only chooser automation; Chromium still owns the real directory handle."""

import ctypes
import os
from pathlib import Path

import dbus
import dbus.service
from dbus.mainloop.glib import DBusGMainLoop
from gi.repository import GLib

DBusGMainLoop(set_as_default=True)
bus = dbus.SessionBus()
name = dbus.service.BusName('org.freedesktop.portal.Desktop', bus=bus)
requests = []


class Request(dbus.service.Object):
    @dbus.service.signal('org.freedesktop.portal.Request', signature='ua{sv}')
    def Response(self, code, results):
        pass

    @dbus.service.method('org.freedesktop.portal.Request')
    def Close(self):
        pass


def allow_write():
    # The isolated Xvfb session contains only the test Chrome. Click its
    # native "Allow this site to edit files?" button after the chooser returns.
    x11 = ctypes.CDLL('libX11.so.6')
    xtst = ctypes.CDLL('libXtst.so.6')
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XFlush.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    xtst.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
    xtst.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
    display = x11.XOpenDisplay(None)
    xtst.XTestFakeMotionEvent(display, -1, 802, 224, 0)
    xtst.XTestFakeButtonEvent(display, 1, 1, 0)
    xtst.XTestFakeButtonEvent(display, 1, 0, 0)
    x11.XFlush(display)
    x11.XCloseDisplay(display)
    return False


class Portal(dbus.service.Object):
    @dbus.service.method('org.freedesktop.DBus.Properties', in_signature='ss', out_signature='v')
    def Get(self, interface, prop):
        return dbus.UInt32(4)

    @dbus.service.method('org.freedesktop.DBus.Properties', in_signature='s', out_signature='a{sv}')
    def GetAll(self, interface):
        return {'version': dbus.UInt32(4)}

    @dbus.service.method(
        'org.freedesktop.portal.FileChooser', in_signature='ssa{sv}', out_signature='o', sender_keyword='sender'
    )
    def OpenFile(self, parent, title, options, sender=None):
        path = '/org/freedesktop/portal/desktop/request/' + sender[1:].replace('.', '_') + '/' + options['handle_token']
        request = Request(bus, path)
        requests.append(request)

        def reply():
            uri = Path(os.environ['LATTI_BROWSER_ARTIFACTS']).as_uri()
            request.Response(0, {'uris': dbus.Array([uri], signature='s')})
            GLib.timeout_add(1200, allow_write)
            return False

        GLib.timeout_add(200, reply)
        return path


portal = Portal(bus, '/org/freedesktop/portal/desktop')
GLib.MainLoop().run()
