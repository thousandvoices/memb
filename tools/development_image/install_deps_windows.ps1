$flatbuffers_version = '1.9.0'

$flatbuffers_url = "https://github.com/google/flatbuffers/archive/v$flatbuffers_version.zip"
$flatc_url = "https://github.com/google/flatbuffers/releases/download/v$flatbuffers_version/flatc_windows_exe.zip"
$flatbuffers_archive = "flatbuffers.zip"
$flatc_archive = "flatc_windows_exe.zip"
$flatbuffers_install_path = "C:\flatbuffers"

Start-FileDownload $flatbuffers_url -FileName $flatbuffers_archive
Expand-Archive $flatbuffers_archive -DestinationPath $flatbuffers_install_path
Remove-Item $flatbuffers_archive

Start-FileDownload $flatc_url -FileName $flatc_archive
Expand-Archive $flatc_archive -DestinationPath $flatbuffers_install_path
Remove-Item $flatc_archive
