#!/usr/bin/env python3
"""The bench's only way out of its private network: an HTTPS proxy that opens tunnels to the hosts it is told
to allow and refuses everything else. Standard library only. Every request is logged as one JSON line on
stdout, allowed or refused, so a run can show what the agent's containers tried to reach.

  python3 egress_proxy.py --allow ai-gateway.vercel.sh:443 --port 3128
"""
import argparse
import asyncio
import json
import time


def log(**kw):
    print(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **kw}), flush=True)


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle(reader, writer, allow):
    peer = writer.get_extra_info("peername")
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 30)
    except Exception:
        writer.close()
        return
    line = head.split(b"\r\n", 1)[0].decode("latin-1")
    parts = line.split()
    if len(parts) != 3 or parts[0] != "CONNECT":
        log(event="refused", reason="only CONNECT is served", request=line[:200], client=str(peer))
        writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()
        return
    target = parts[1].lower()
    if target not in allow:
        log(event="refused", reason="host not allowed", target=target, client=str(peer))
        writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()
        return
    host, port = target.rsplit(":", 1)
    try:
        up_r, up_w = await asyncio.wait_for(asyncio.open_connection(host, int(port)), 30)
    except Exception as e:
        log(event="failed", target=target, error=str(e)[:200], client=str(peer))
        writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()
        return
    log(event="allowed", target=target, client=str(peer))
    writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    await writer.drain()
    await asyncio.gather(pipe(reader, up_w), pipe(up_r, writer))


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow", action="append", required=True, help="host:port to allow; repeat for several")
    ap.add_argument("--port", type=int, default=3128)
    args = ap.parse_args()
    allow = {a.lower() for a in args.allow}
    server = await asyncio.start_server(lambda r, w: handle(r, w, allow), "0.0.0.0", args.port)
    log(event="listening", port=args.port, allow=sorted(allow))
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
