set(requires
    hardware_driver
    sr_ringbuf
    esp_wifi
    esp_event
    esp_netif
    nvs_flash
    )

idf_component_register(SRCS main.c
                       REQUIRES ${requires})

