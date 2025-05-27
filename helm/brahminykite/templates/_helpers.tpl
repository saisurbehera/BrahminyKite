{{/*
Expand the name of the chart.
*/}}
{{- define "brahminykite.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "brahminykite.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "brahminykite.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "brahminykite.labels" -}}
helm.sh/chart: {{ include "brahminykite.chart" . }}
{{ include "brahminykite.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "brahminykite.selectorLabels" -}}
app.kubernetes.io/name: {{ include "brahminykite.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "brahminykite.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "brahminykite.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Get the namespace
*/}}
{{- define "brahminykite.namespace" -}}
{{- if .Values.namespace.create }}
{{- .Values.namespace.name | default (include "brahminykite.fullname" .) }}
{{- else }}
{{- .Release.Namespace }}
{{- end }}
{{- end }}

{{/*
Return the appropriate apiVersion for deployment.
*/}}
{{- define "brahminykite.deployment.apiVersion" -}}
{{- print "apps/v1" -}}
{{- end -}}

{{/*
Return the appropriate apiVersion for ingress.
*/}}
{{- define "brahminykite.ingress.apiVersion" -}}
{{- print "networking.k8s.io/v1" -}}
{{- end -}}

{{/*
Get the PostgreSQL hostname
*/}}
{{- define "brahminykite.postgresql.fullname" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-postgresql" (include "brahminykite.fullname" .) -}}
{{- else }}
{{- .Values.externalPostgresql.host -}}
{{- end }}
{{- end -}}

{{/*
Get the Redis hostname
*/}}
{{- define "brahminykite.redis.fullname" -}}
{{- if .Values.redis.enabled }}
{{- printf "%s-redis-master" (include "brahminykite.fullname" .) -}}
{{- else }}
{{- .Values.externalRedis.host -}}
{{- end }}
{{- end -}}

{{/*
Get the PostgreSQL port
*/}}
{{- define "brahminykite.postgresql.port" -}}
{{- if .Values.postgresql.enabled }}
{{- .Values.postgresql.primary.service.ports.postgresql | default 5432 }}
{{- else }}
{{- .Values.externalPostgresql.port | default 5432 }}
{{- end }}
{{- end -}}

{{/*
Get the Redis port
*/}}
{{- define "brahminykite.redis.port" -}}
{{- if .Values.redis.enabled }}
{{- .Values.redis.master.service.ports.redis | default 6379 }}
{{- else }}
{{- .Values.externalRedis.port | default 6379 }}
{{- end }}
{{- end -}}

{{/*
Get the PostgreSQL user
*/}}
{{- define "brahminykite.postgresql.username" -}}
{{- if .Values.postgresql.enabled }}
{{- .Values.global.postgresql.auth.username | default .Values.postgresql.auth.username }}
{{- else }}
{{- .Values.externalPostgresql.username }}
{{- end }}
{{- end -}}

{{/*
Get the PostgreSQL password secret
*/}}
{{- define "brahminykite.postgresql.secretName" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-postgresql" (include "brahminykite.fullname" .) -}}
{{- else }}
{{- .Values.externalPostgresql.existingSecret }}
{{- end }}
{{- end -}}

{{/*
Get the PostgreSQL password secret key
*/}}
{{- define "brahminykite.postgresql.secretKey" -}}
{{- if .Values.postgresql.enabled }}
{{- "password" }}
{{- else }}
{{- .Values.externalPostgresql.existingSecretPasswordKey }}
{{- end }}
{{- end -}}

{{/*
Get the Redis password secret
*/}}
{{- define "brahminykite.redis.secretName" -}}
{{- if and .Values.redis.enabled .Values.redis.auth.enabled }}
{{- printf "%s-redis" (include "brahminykite.fullname" .) -}}
{{- else if .Values.externalRedis.existingSecret }}
{{- .Values.externalRedis.existingSecret }}
{{- end }}
{{- end -}}

{{/*
Create image pull secret
*/}}
{{- define "brahminykite.imagePullSecret" }}
{{- with .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- range . }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Return the proper service image name
*/}}
{{- define "brahminykite.service.image" -}}
{{- $registryName := .Values.services.image.registry -}}
{{- $repositoryName := .service.image.repository -}}
{{- $tag := .service.image.tag | default .Values.services.image.tag | default .Chart.AppVersion -}}
{{- printf "%s/%s:%s" $registryName $repositoryName $tag -}}
{{- end -}}

{{/*
Return the proper API image name
*/}}
{{- define "brahminykite.api.image" -}}
{{- $registryName := .Values.api.image.registry -}}
{{- $repositoryName := .Values.api.image.repository -}}
{{- $tag := .Values.api.image.tag | default .Chart.AppVersion -}}
{{- printf "%s/%s:%s" $registryName $repositoryName $tag -}}
{{- end -}}

{{/*
Create a default service configuration
*/}}
{{- define "brahminykite.service.env" -}}
- name: SERVICE_NAME
  value: {{ .serviceName | quote }}
- name: SERVICE_PORT
  value: {{ .servicePort | quote }}
- name: METRICS_PORT
  value: {{ .Values.monitoring.metrics.port | default "9090" | quote }}
- name: LOG_LEVEL
  value: {{ .Values.logging.level | default "INFO" | quote }}
- name: LOG_FORMAT
  value: {{ .Values.logging.format | default "json" | quote }}
- name: POSTGRES_HOST
  value: {{ include "brahminykite.postgresql.fullname" . | quote }}
- name: POSTGRES_PORT
  value: {{ include "brahminykite.postgresql.port" . | quote }}
- name: POSTGRES_DB
  value: {{ .Values.global.postgresql.auth.database | quote }}
- name: POSTGRES_USER
  value: {{ include "brahminykite.postgresql.username" . | quote }}
- name: POSTGRES_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "brahminykite.postgresql.secretName" . }}
      key: {{ include "brahminykite.postgresql.secretKey" . }}
- name: REDIS_HOST
  value: {{ include "brahminykite.redis.fullname" . | quote }}
- name: REDIS_PORT
  value: {{ include "brahminykite.redis.port" . | quote }}
{{- if and .Values.redis.auth.enabled .Values.redis.enabled }}
- name: REDIS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "brahminykite.redis.secretName" . }}
      key: redis-password
{{- end }}
{{- if .Values.monitoring.tracing.enabled }}
- name: TRACING_ENABLED
  value: "true"
- name: TRACING_ENDPOINT
  value: {{ .Values.monitoring.tracing.endpoint | quote }}
- name: TRACING_SAMPLING_RATE
  value: {{ .Values.monitoring.tracing.samplingRate | quote }}
{{- end }}
{{- if .Values.monitoring.profiling.enabled }}
- name: PROFILING_ENABLED
  value: "true"
- name: PROFILING_PORT
  value: {{ .Values.monitoring.profiling.port | quote }}
{{- end }}
{{- end -}}

{{/*
Pod annotations
*/}}
{{- define "brahminykite.podAnnotations" -}}
{{- if .Values.monitoring.metrics.enabled }}
prometheus.io/scrape: "true"
prometheus.io/port: {{ .Values.monitoring.metrics.port | default "9090" | quote }}
prometheus.io/path: {{ .Values.monitoring.metrics.path | default "/metrics" | quote }}
{{- end }}
{{- with .Values.podAnnotations }}
{{- toYaml . }}
{{- end }}
{{- end -}}

{{/*
Security context for containers
*/}}
{{- define "brahminykite.containerSecurityContext" -}}
{{- if .Values.containerSecurityContext }}
securityContext:
  {{- toYaml .Values.containerSecurityContext | nindent 2 }}
{{- end }}
{{- end -}}

{{/*
Security context for pods
*/}}
{{- define "brahminykite.podSecurityContext" -}}
{{- if .Values.podSecurityContext }}
securityContext:
  {{- toYaml .Values.podSecurityContext | nindent 2 }}
{{- end }}
{{- end -}}