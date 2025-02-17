To run 

```bash
chmod +x ./start.sh
```

```bash
./start.sh
```

or 

```bash
pip install -r requirements.txt
```

```bash
python app.py
```

Supabase Integration
open one more terminal and enter:

```bash
 curl http://127.0.0.1:8001/test-supabase
 ```
 ```bash
 curl -X POST "http://127.0.0.1:8001/process-message/"      -H "Content-Type: application/json"      -d '{"session_id": "test-session-123", "content": "Hello, AI!"}''
 ```

