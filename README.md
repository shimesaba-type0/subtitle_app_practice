


## How to generate secret key

```py
import secrets

# 32 バイトのランダムなURLセーフな文字列を生成
secret_key = secrets.token_urlsafe(32)
print(secret_key)
```

