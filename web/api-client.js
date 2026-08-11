"use strict";

async function handle(response) {
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json")
    ? await response.json()
    : { error: await response.text() };
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `Richiesta fallita (${response.status})`);
  }
  return data;
}

export async function getJSON(path) {
  const response = await fetch(path, { method: "GET" });
  return handle(response);
}

export async function postJSON(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  return handle(response);
}

export async function postForm(path, file) {
  const form = new FormData();
  form.append("file", file, file.name);
  const response = await fetch(path, { method: "POST", body: form });
  return handle(response);
}
