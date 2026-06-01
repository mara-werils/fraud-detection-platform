"""Tests for PII masking utilities."""

from shared.pii import mask_card_numbers, mask_dict, mask_emails, mask_value


class TestMaskValue:
    def test_short_string_fully_masked(self):
        assert mask_value("abc") == "***"

    def test_long_string_preserves_last_four(self):
        result = mask_value("4111111111111111")
        assert result.endswith("1111")
        assert result.startswith("*")

    def test_visible_chars_configurable(self):
        result = mask_value("secret_value", visible_chars=2)
        assert result.endswith("ue")


class TestMaskDict:
    def test_sensitive_fields_masked(self):
        data = {"user": "alice", "password": "hunter2", "email": "alice@example.com"}
        result = mask_dict(data)
        assert result["user"] == "alice"
        assert "hunter2" not in result["password"]
        assert "alice@example.com" not in result["email"]

    def test_nested_dicts(self):
        data = {"outer": {"api_key": "sk-12345"}}
        result = mask_dict(data)
        assert "sk-12345" not in result["outer"]["api_key"]

    def test_none_values_preserved(self):
        data = {"password": None}
        result = mask_dict(data)
        assert result["password"] is None


class TestMaskCardNumbers:
    def test_masks_card_in_text(self):
        text = "Charged card 4111 1111 1111 1111 successfully"
        result = mask_card_numbers(text)
        assert "4111 1111 1111 1111" not in result

    def test_no_card_unchanged(self):
        text = "No card here"
        assert mask_card_numbers(text) == text


class TestMaskEmails:
    def test_masks_email_in_text(self):
        text = "Contact alice@example.com for details"
        result = mask_emails(text)
        assert "alice@example.com" not in result
