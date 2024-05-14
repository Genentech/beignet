from prescient.io._parse_s3_path import parse_s3_path


def test_parse_s3_path():
    s3_path = "s3://my-bucket/my-folder/my-file.txt"
    expected_bucket_name = "my-bucket"
    expected_bucket_key = "my-folder/my-file.txt"

    bucket_name, bucket_key = parse_s3_path(s3_path)

    assert bucket_name == expected_bucket_name
    assert bucket_key == expected_bucket_key
