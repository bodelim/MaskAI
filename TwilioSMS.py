# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client


# Your Account Sid and Auth Token from twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = 'ACd30462945303ace7ecab19130390413f'
auth_token = '3782692238f888911572efe1b530ab95'
client = Client(account_sid, auth_token)

message = client.messages \
                .create(
                     body="테스트 문자전송",
                     from_='+13158093845',
                     to='+821074517300'
                 )

print(message.sid)
