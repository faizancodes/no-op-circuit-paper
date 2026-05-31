    def location(self, item):
        return item.get_absolute_url()

    def get_protocol(self, protocol=None):
        # Determine protocol
        if self.protocol is None and protocol is None:
            warnings.warn(
                "The default sitemap protocol will be changed from 'http' to "
                "'https' in Django 5.0. Set Sitemap.protocol to silence this "
                "warning.",
                category=RemovedInDjango50Warning,
                stacklevel=2,
            )
        # RemovedInDjango50Warning: when the deprecation ends, replace 'http'
        # with 'https'.
        return self.protocol or protocol or "http"

    def get_domain(self, site=None):
        # Determine domain
        if site is None:
            if django_apps.is_installed("django.contrib.sites"):
                Site = django_apps.get_model("sites.Site")
                try:
                    site = Site.objects.get_current()
                except Site.DoesNotExist:
                    pass
            if site is None:
                raise ImproperlyConfigured(
                    "To use sitemaps, either enable the sites framework or pass "
                    "a Site/RequestSite object in your view."
                )
        return site.domain

    def get_urls(self, page=1, site=None, protocol=None):
        protocol = self.get_protocol(protocol)
        domain = self.get_domain(site)
        return self._urls(page, protocol, domain)

    def get_latest_lastmod(self):
        if not hasattr(self, "lastmod"):
            return None
        if callable(self.lastmod):
            try:
                return max([self.lastmod(item) for item in self.items()], default=None)
            except TypeError:
                return None
        else:
            return self.lastmod

    def _urls(self, page, protocol, domain):
        urls = []
        latest_lastmod = None
        all_items_lastmod = True  # track if all items have a lastmod

        paginator_page = self.paginator.page(page)
        for item in paginator_page.object_list:
            loc = f"{protocol}://{domain}{self._location(item)}"
            priority = self._get("priority", item)
            lastmod = self._get("lastmod", item)

            if all_items_lastmod:
                all_items_lastmod = lastmod is not None
                if all_items_lastmod and (
                    latest_lastmod is None or lastmod > latest_lastmod
                ):
                    latest_lastmod = lastmod

            url_info = {
                "item": item,
                "location": loc,
                "lastmod": lastmod,
                "changefreq": self._get("changefreq", item),
                "priority": str(priority if priority is not None else ""),
                "alternates": [],
            }

            if self.i18n and self.alternates:
                for lang_code in self._languages():
                    loc = f"{protocol}://{domain}{self._location(item, lang_code)}"
                    url_info["alternates"].append(
                        {
                            "location": loc,
                            "lang_code": lang_code,
                        }
                    )
                if self.x_default:
                    lang_code = settings.LANGUAGE_CODE
