$flatbuffers_version = '1.9.0'

$flatbuffers_url = "https://github.com/google/flatbuffers/releases/download/v$flatbuffers_version/flatc_windows_exe.zip"
$flatbuffers_archive = "flatc_windows_exe.zip"
$flatbuffers_install_path = "C:\flatbuffers"

Start-FileDownload $flatbuffers_url -FileName $flatbuffers_archive
Expand-Archive $flatbuffers_archive -DestinationPath $flatbuffers_install_path
