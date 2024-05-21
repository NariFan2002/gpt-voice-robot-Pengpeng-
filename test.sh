
curl https://api.chatanywhere.tech/v1/chat/completions \
 -H 'Content-Type: application/json' \
 -H 'Authorization: Bearer sk-yb1BsOYZPsz18D7njNak05kJUFnHrUKfX78Xo2D19ujP75xl' \
 -d '{
 "model": "gpt-3.5-turbo",
 "messages": [{"role": "user", "content": "Say this is a test!"}],
 "temperature": 0.7
}'
