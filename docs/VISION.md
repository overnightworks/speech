# speech-runtime vision

## What it does

A speech service that gives any app a voice and an ear. It turns text into
natural speech and speech into text with AI models, behind one interface that
follows the common standard for speech APIs, so an app built against that
standard works with it unchanged.

- **Speaking:** text in, natural speech out, in the voice the app picks from
  those on offer.
- **Hearing:** a recording or a live audio stream in, the words out as they are
  spoken.
- **Interchangeable models:** new voices and hearing models join by
  configuration, and no app has to change for it.
- **Trusted callers only:** it serves the apps it knows, each with everything it
  offers, and says clearly when it cannot serve a request.

## Who uses it today

presentator and hopin are registered as its callers. presentator's
co-presenter speaks and hears through it live; hopin's game master speaks
through it.

## Deliberate non-goals

- It does not understand or answer; it speaks and hears, and what is said
  belongs to the app.
- It knows nothing about its callers beyond which apps are registered.
- It keeps nothing it hears or speaks.
