        parser.add_argument(
            "args",
            metavar="app_label",
            nargs="*",
            help="Specify the app label(s) to create migrations for.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Just show what migrations would be made; don't actually write them.",
        )
        parser.add_argument(
            "--merge",
            action="store_true",
            help="Enable fixing of migration conflicts.",
        )
        parser.add_argument(
            "--empty",
            action="store_true",
            help="Create an empty migration.",
        )
        parser.add_argument(
            "--noinput",
            "--no-input",
            action="store_false",
            dest="interactive",
            help="Tells Django to NOT prompt the user for input of any kind.",
        )
        parser.add_argument(
            "-n",
            "--name",
            help="Use this name for migration file(s).",
        )
        parser.add_argument(
            "--no-header",
            action="store_false",
            dest="include_header",
            help="Do not add header comments to new migration file(s).",
        )
        parser.add_argument(
            "--check",
            action="store_true",
            dest="check_changes",
            help="Exit with a non-zero status if model changes are missing migrations.",
        )
        parser.add_argument(
            "--scriptable",
            action="store_true",
            dest="scriptable",
            help=(
                "Divert log output and input prompts to stderr, writing only "
                "paths of generated migration files to stdout."
            ),
        )
        parser.add_argument(
            "--update",
            action="store_true",
            dest="update",
            help=(
                "Merge model changes into the latest migration and optimize the "
                "resulting operations."
            ),
        )

    @property
    def log_output(self):
        return self.stderr if self.scriptable else self.stdout

    def log(self, msg):
        self.log_output.write(msg)

    @no_translations
    def handle(self, *app_labels, **options):
        self.written_files = []
        self.verbosity = options["verbosity"]
        self.interactive = options["interactive"]
        self.dry_run = options["dry_run"]
        self.merge = options["merge"]
        self.empty = options["empty"]
        self.migration_name = options["name"]
        if self.migration_name and not self.migration_name.isidentifier():
            raise CommandError("The migration name must be a valid Python identifier.")
        self.include_header = options["include_header"]
        check_changes = options["check_changes"]
        self.scriptable = options["scriptable"]
        self.update = options["update"]
        # If logs and prompts are diverted to stderr, remove the ERROR style.
