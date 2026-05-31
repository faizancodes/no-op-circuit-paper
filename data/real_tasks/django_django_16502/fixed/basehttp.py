
    def close_request(self, request):
        self._close_connections()
        super().close_request(request)


class ServerHandler(simple_server.ServerHandler):
    http_version = "1.1"

    def __init__(self, stdin, stdout, stderr, environ, **kwargs):
        """
        Use a LimitedStream so that unread request data will be ignored at
        the end of the request. WSGIRequest uses a LimitedStream but it
        shouldn't discard the data since the upstream servers usually do this.
        This fix applies only for testserver/runserver.
        """
        try:
            content_length = int(environ.get("CONTENT_LENGTH"))
        except (ValueError, TypeError):
            content_length = 0
        super().__init__(
            LimitedStream(stdin, content_length), stdout, stderr, environ, **kwargs
        )

    def cleanup_headers(self):
        super().cleanup_headers()
        # HTTP/1.1 requires support for persistent connections. Send 'close' if
        # the content length is unknown to prevent clients from reusing the
        # connection.
        if "Content-Length" not in self.headers:
            self.headers["Connection"] = "close"
        # Persistent connections require threading server.
        elif not isinstance(self.request_handler.server, socketserver.ThreadingMixIn):
            self.headers["Connection"] = "close"
        # Mark the connection for closing if it's set as such above or if the
        # application sent the header.
        if self.headers.get("Connection") == "close":
            self.request_handler.close_connection = True

    def close(self):
        self.get_stdin().read()
        super().close()

    def finish_response(self):
        if self.environ["REQUEST_METHOD"] == "HEAD":
            try:
                deque(self.result, maxlen=0)  # Consume iterator.
                # Don't call self.finish_content() as, if the headers have not
                # been sent and Content-Length isn't set, it'll default to "0"
                # which will prevent omission of the Content-Length header with
                # HEAD requests as permitted by RFC 9110 Section 9.3.2.
                # Instead, send the headers, if not sent yet.
                if not self.headers_sent:
                    self.send_headers()
            finally:
                self.close()
        else:
            super().finish_response()


class WSGIRequestHandler(simple_server.WSGIRequestHandler):
    protocol_version = "HTTP/1.1"

    def address_string(self):
        # Short-circuit parent method to not call socket.getfqdn
        return self.client_address[0]

    def log_message(self, format, *args):
        extra = {
            "request": self.request,
            "server_time": self.log_date_time_string(),
        }
        if args[1][0] == "4":
            # 0x16 = Handshake, 0x03 = SSL 3.0 or TLS 1.x
            if args[0].startswith("\x16\x03"):
                extra["status_code"] = 500
                logger.error(
                    "You're accessing the development server over HTTPS, but "
                    "it only supports HTTP.",
                    extra=extra,
                )
                return

        if args[1].isdigit() and len(args[1]) == 3:
            status_code = int(args[1])
            extra["status_code"] = status_code

            if status_code >= 500:
                level = logger.error
            elif status_code >= 400:
                level = logger.warning
            else:
                level = logger.info
        else:
            level = logger.info

        level(format, *args, extra=extra)

    def get_environ(self):
        # Strip all headers with underscores in the name before constructing
        # the WSGI environ. This prevents header-spoofing based on ambiguity
        # between underscores and dashes both normalized to underscores in WSGI
