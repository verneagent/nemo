#!/usr/bin/env node
// TP-Link EasyMesh Router CLI
// Usage: node tplink.js -p <password> [status|devices|raw]

const CONFIG = { ip: process.env.TPLINK_IP || '192.168.0.1' };
const KEY1 = 'RDpbLfCPsJZ7fiv';
const KEY2 = 'yLwVl0zKqws7LgKPRQ84Mdt708T1qQ3Ha7xv3H7NyU84p21BriUWBU43odz3iP4rBL3cD02KZciXTysVXiV8ngg6vL48rPJyAUw0HurW20xqxv9aYb4M9wK1Ae0wlro510qXeU07kV57fQMc8L6aLgMLwygtc0F10a0Dg70TOoouyFhdysuRMO51yY5ZlOZZLEal1h0t9YQW0Ko7oBwmCAHoic4HYbUyVeU3sfQ1xtXcPcf1aT303wAQhv66qzW';

// ---- Auth ----
function securityEncode(input, key1, key2) {
  let result = '';
  const maxLen = Math.max(input.length, key1.length);
  for (let m = 0; m < maxLen; m++) {
    let k = 187, l = 187;
    if (m >= input.length) l = key1.charCodeAt(m);
    else if (m >= key1.length) k = input.charCodeAt(m);
    else { k = input.charCodeAt(m); l = key1.charCodeAt(m); }
    result += key2.charAt((k ^ l) % key2.length);
  }
  return result;
}

async function login(pwd) {
  const encoded = securityEncode(pwd, KEY1, KEY2);
  const resp = await fetch(`http://${CONFIG.ip}/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json; charset=UTF-8', 'Referer': `http://${CONFIG.ip}/` },
    body: JSON.stringify({ method: 'do', login: { password: encoded } }),
  });
  const data = await resp.json();
  if (data.error_code !== 0) throw new Error(`Login failed: ${JSON.stringify(data)}`);
  // Stok is already URL-encoded; decode then re-encode properly
  return encodeURIComponent(decodeURIComponent(data.stok));
}

async function api(stok, query) {
  const url = `http://${CONFIG.ip}/stok=${stok}/ds`;
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json; charset=UTF-8', 'Referer': `http://${CONFIG.ip}/` },
    body: JSON.stringify({ ...query, method: 'get' }),
  });
  return resp.json();
}

// ---- Commands ----
async function cmdStatus(stok) {
  const r = await api(stok, {
    system: { name: ['sys'] },
    hosts_info: { table: 'host_info' },
    network: { name: 'iface_mac' },
    function: { name: 'new_module_spec' },
  });

  const spec = r.function?.new_module_spec || {};
  const net = r.network?.iface_mac || {};
  const hosts = Object.values(r.hosts_info?.host_info?.[0] ? r.hosts_info.host_info : {})
    .map(h => Object.values(h)[0]);

  console.log('=== Router ===');
  console.log(`  Eth bandwidth: ${spec.eth_bandwidth || '?'} Mbps`);
  console.log(`  Band steering: ${spec.wifison === '1' ? 'ON' : 'OFF'}`);
  console.log(`  Mesh: ${spec.wifison_mesh === '1' ? 'active' : 'inactive'} (role: ${spec.wifison_role || '?'})`);
  console.log(`  HW NAT: ${spec.hnat === '1' ? 'enabled' : 'disabled'}`);
  console.log(`  App version: ${spec.app_version || '?'}`);

  console.log(`\n=== Devices (${hosts.length}) ===`);
  const bandMap = { '0': '2.4G', '1': '5G', '2': 'ETH' };
  for (const h of hosts) {
    const band = bandMap[h.wifi_mode] || '?';
    const name = decodeURIComponent((h.hostname || '').replace(/\+/g, ' ')) || '(unnamed)';
    const mesh = h.is_mesh === '1' ? ' [MESH]' : '';
    console.log(`  ${name.padEnd(35)} ${band.padEnd(5)} ${h.ip?.padEnd(16) || '-'.padEnd(16)} ${h.mac}${mesh}`);
  }
}

async function main() {
  const args = process.argv.slice(2);
  let pwd = process.env.TPLINK_PWD, i = 0;
  if (args[0] === '-p') { pwd = args[1]; i = 2; }
  if (!pwd) { console.error('Usage: node tplink.js -p <password> [status|devices|raw]'); process.exit(1); }

  const stok = await login(pwd);
  const cmd = args[i] || 'status';

  switch (cmd) {
    case 'status': case 'devices': await cmdStatus(stok); break;
    case 'raw': {
      const r = await api(stok, JSON.parse(args[i+1] || '{"system":{"name":["sys"]}}'));
      console.log(JSON.stringify(r, null, 2));
      break;
    }
    default: console.error(`Unknown: ${cmd}`); process.exit(1);
  }
}
main().catch(e => { console.error('Error:', e.message); process.exit(1); });
