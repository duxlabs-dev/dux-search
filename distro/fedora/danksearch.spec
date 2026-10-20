# Spec for Dux Search - pre-built binary from GitHub releases

%global debug_package %{nil}
%global pkg_summary Blazingly fast and efficient file system search tool

Name:           danksearch
Version:        0.0.7
Release:        1%{?dist}
Summary:        %{pkg_summary}

License:        MIT
URL:            https://duxlabs.dev/docs/dsearch/
VCS:            https://github.com/duxlabs-dev/dux-search

BuildRequires:  wget
BuildRequires:  gzip
BuildRequires:  coreutils
BuildRequires:  systemd-rpm-macros

Requires:       glibc

%description
Dux Search is a file system search utility designed for the Dux Labs modern
desktop suite. It provides rapid filesystem searching capabilities optimized
for performance and efficiency. The tool integrates seamlessly with
Dux Shell and its launcher system, enabling users to quickly locate
files across their system.

Powered by the bleve search library, Dux Search supports fuzzy search, EXIF
extraction, virtual folders, and concurrent indexing for blazingly fast results.

%prep
# Download and extract Dux Search binary for target architecture
case "%{_arch}" in
  x86_64)
    DSEARCH_ARCH="amd64"
    ;;
  aarch64)
    DSEARCH_ARCH="arm64"
    ;;
  *)
    echo "Unsupported architecture: %{_arch}"
    exit 1
    ;;
esac

wget -O %{_builddir}/dsearch.gz "https://github.com/duxlabs-dev/dux-search/releases/latest/download/dsearch-linux-${DSEARCH_ARCH}.gz" || {
  echo "Failed to download dsearch for architecture %{_arch}"
  exit 1
}
gunzip -c %{_builddir}/dsearch.gz > %{_builddir}/dsearch
chmod +x %{_builddir}/dsearch

# Create systemd user service file inline for RPM installation
cat > %{_builddir}/dsearch.service << 'EOF'
[Unit]
Description=dsearch - Fast filesystem search service
Documentation=https://github.com/duxlabs-dev/dux-search
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/dsearch serve
Restart=on-failure
RestartSec=5s
TimeoutStopSec=30s

StandardOutput=journal
StandardError=journal
SyslogIdentifier=dsearch

[Install]
WantedBy=default.target
EOF

%build
# Using pre-built binary - nothing to build

%install
# Install dsearch binary
install -Dm755 %{_builddir}/dsearch %{buildroot}%{_bindir}/dsearch

# Install systemd user service
install -Dm644 %{_builddir}/dsearch.service %{buildroot}%{_userunitdir}/dsearch.service

%files
%{_bindir}/dsearch
%{_userunitdir}/dsearch.service

%post
# Initial install setup
if [ "$1" -eq 1 ]; then
cat << 'EOF'

=========================================================================
        Thanks for installing Dux Search!
=========================================================================

To configure, enable the systemd user unit service:

    systemctl --user enable --now dsearch

For more documentation visit:

    https://duxlabs.dev/docs/dsearch

=========================================================================

EOF
fi

%changelog
* Fri Nov 1 2025 DankLinux Team <noreply@danklinux.com> - 0.0.7-1
- Update to v0.0.7
- Use /latest/ for automatic updates
- Pre-built binary from GitHub releases
- Includes systemd user service for autostart
